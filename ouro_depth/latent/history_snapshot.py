"""Fixed-trace S6 history snapshot: detached prompt rows plus response cache leaves.

A snapshot is immutable during K-hop replay and expires with the optimizer step.
Rows are the packed ``register.py::pack`` layout, one [1, prompt+n-1, R] tensor
per layer, collected under no-grad or exported from the same-version vLLM rollout.
"""
from dataclasses import dataclass

import torch

from .batched_engine import BatchedRollingEngine

SNAPSHOT_SCHEMA = "s6-khop-snapshot-v1"


@dataclass
class HistorySnapshot:
    rows: tuple                 # per-layer packed [1, prompt+response-1, R]
    positions: torch.Tensor     # absolute [prompt+response-1] positions, contiguous
    prompt_length: int
    response_length: int
    first_response_logits: torch.Tensor  # detached [1, 1, vocab]
    dtype: torch.dtype
    source: str = "reference_collect"
    schema: str = SNAPSHOT_SCHEMA


@torch.no_grad()
def collect_snapshot(model, student, ids, prompt, *, serving_numerics=False):
    """Serial no-grad C1 pass; per-step detach moves rows into doubling storage."""
    # A no-grad prefill inside the outer autocast scope must not populate its
    # weight-cast cache with detached copies later reused by differentiable replay.
    device_type = ids.device.type
    with torch.autocast(device_type, enabled=torch.is_autocast_enabled(device_type),
                        dtype=torch.get_autocast_dtype(device_type), cache_enabled=False):
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= prompt < ids.shape[1]:
            raise ValueError('Expected [1,L] IDs with a nonempty prompt and response')
        n = ids.shape[1] - prompt
        engine = BatchedRollingEngine(model, student, False, serving_numerics=serving_numerics)
        first, _ = engine.prefill(ids[:, :prompt], chunk_size=prompt, last_logits_only=True)
        engine.detach_history()
        for position in range(prompt, ids.shape[1] - 1):
            engine.step(ids[:, position:position + 1], emit_logits=False)
            engine.detach_history()
        length = prompt + n - 1
        rows = tuple(row[:, :length].clone() for row in engine.prefix)
        snapshot = HistorySnapshot(rows=rows, positions=torch.arange(length, device=ids.device),
                                   prompt_length=prompt, response_length=n,
                                   first_response_logits=first.detach().clone(),
                                   dtype=rows[0].dtype if rows else None)
        validate_snapshot(snapshot, student)
        return snapshot


def validate_snapshot(snapshot, student):
    cfg = student.cfg
    if snapshot.schema != SNAPSHOT_SCHEMA or snapshot.source not in ("reference_collect", "vllm_rollout"):
        raise ValueError('Unsupported history snapshot schema or source')
    if snapshot.prompt_length < 1 or snapshot.response_length < 1:
        raise ValueError('Snapshot prompt/response must be nonempty')
    length = snapshot.prompt_length + snapshot.response_length - 1
    width = cfg['rank'] + cfg['rank_v'] + 2 * cfg['rank1']
    if len(snapshot.rows) != cfg['num_layers']:
        raise ValueError('Snapshot layer count differs from the student')
    for row in snapshot.rows:
        if row.shape != (1, length, width):
            raise ValueError('Snapshot row layout differs from packed S6 ranks')
        if row.dtype != snapshot.dtype:
            raise ValueError('Snapshot dtype is not uniform across layers')
    if snapshot.positions.shape != (length,) or \
            not bool((snapshot.positions == torch.arange(length, device=snapshot.positions.device)).all()):
        raise ValueError('Snapshot positions are not contiguous absolute indices')
    if snapshot.first_response_logits.ndim != 3 or snapshot.first_response_logits.shape[:2] != (1, 1):
        raise ValueError('Snapshot must keep exactly the first response logits')
    return snapshot


def load_rollout_snapshot(trajectory, student):
    """Load one request only; metadata guards against stale or misaligned cache."""
    ref = trajectory.history_ref
    if not ref or ref.get('version') != trajectory.version or ref.get('request_id') != trajectory.request_id:
        raise ValueError('Missing, stale or mismatched rollout cache reference')
    data = torch.load(ref['path'], map_location='cpu', weights_only=True)
    length = trajectory.ids.shape[1]-1
    if data.get('schema') != 's6-rollout-cache-v1' or data.get('version') != trajectory.version or \
            data.get('request_id') != trajectory.request_id or data.get('cfg') != student.cfg:
        raise ValueError('Rollout snapshot schema/version/request/config mismatch')
    if data.get('length') != length or ref.get('length') != length:
        raise ValueError(f'Rollout cache length mismatch: exported {data.get("length")}, '
            f'reference {ref.get("length")}, trajectory needs {length} '
            f'(request {trajectory.request_id}, prompt {trajectory.prompt}, '
            f'response {trajectory.response_length}, truncated {trajectory.truncated})')
    if data.get('prompt_ids') != trajectory.ids[0, :trajectory.prompt].tolist():
        raise ValueError(f'Rollout cache prompt tokens differ from the trajectory '
                         f'(request {trajectory.request_id})')
    if ref.get('token_ids') != trajectory.ids[0].tolist():
        raise ValueError(f'Rollout cache token ids differ from the trajectory '
                         f'(request {trajectory.request_id})')
    device = trajectory.ids.device
    rows = tuple(row.to(device) for row in data['rows'])
    snapshot = HistorySnapshot(rows=rows, positions=torch.arange(length, device=device),
        prompt_length=trajectory.prompt, response_length=trajectory.response_length,
        first_response_logits=data['first_response_logits'].to(device),
        dtype=rows[0].dtype, source='vllm_rollout')
    return validate_snapshot(snapshot, student)
