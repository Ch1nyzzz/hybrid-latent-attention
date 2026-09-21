"""Cache ownership, physical indexing, retirement order and fail-closed replay."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from ouro_depth.vllm_latent.cache_export import RolloutCacheExporter
from ouro_depth.latent.decode_training import Trajectory
from ouro_depth.latent.history_snapshot import load_rollout_snapshot


def fake_runner():
    caches = [torch.arange(3*1*2*w).reshape(3, 1, 2, w).float() for w in (4, 2)]
    attns = [NS(layer_name=str(i), kv_cache=c) for i, c in enumerate(caches)]
    cfg = dict(num_layers=1, rank=2, rank_v=2, rank1=1)
    body = NS(latent_cfg=cfg, layers=[NS(self_attn=NS(attn_main=attns[0], attn_l1=attns[1]))])
    tables = [NS(num_blocks_per_row=[2], block_size=2, get_cpu_tensor=lambda: torch.tensor([[2, 0]])),
              NS(num_blocks_per_row=[2], block_size=2, get_cpu_tensor=lambda: torch.tensor([[1, 2]]))]
    batch = NS(req_id_to_index={'a': 0}, req_ids=['a'], num_computed_tokens_cpu=[0], block_table=tables)
    runner = NS(get_model=lambda: NS(model=body), speculative_config=None,
        kv_cache_config=NS(kv_cache_groups=[NS(layer_names=['0']), NS(layer_names=['1'])]),
        input_batch=batch, requests={'a': NS(prompt_token_ids=[4, 5])})
    def update(schedule):
        if schedule.finished_req_ids:
            for c in caches:c.fill_(-999)  # simulate block reuse inside original runner
    runner._update_states = update
    def execute(schedule):
        runner._update_states(schedule)
        runner.execute_model_state = NS(logits=torch.arange(9).float()[None] + runner.input_batch.num_computed_tokens_cpu[0])
    runner.execute_model = execute
    return runner, body, caches


def schedule(count=2, finished=()):
    return NS(finished_req_ids=set(finished), num_scheduled_tokens={'a': count} if count else {},
              total_num_scheduled_tokens=count)


def test_export_before_block_reuse_and_final_flush(tmp_path):
    runner, body, caches = fake_runner()
    expected = torch.cat([caches[0][[2, 2, 0], 0, [0, 1, 0]],
                          caches[1][[1, 1, 2], 0, [0, 1, 0]]], -1)[None]
    exporter = RolloutCacheExporter(runner)
    exporter.begin(tmp_path/'v0', 0)
    runner.execute_model(schedule())
    runner.input_batch.num_computed_tokens_cpu[0] = 2
    runner.execute_model(schedule(1))
    runner._update_states(schedule(0, ['a']))
    reply = exporter.finish()
    ref = reply['snapshots']['a']
    ref['token_ids'] = [4, 5, 6, 7]
    trajectory = Trajectory(torch.tensor([[4, 5, 6, 7]]), 2, 0, history_ref=ref, request_id='a')
    snapshot = load_rollout_snapshot(trajectory, NS(cfg=body.latent_cfg))
    torch.testing.assert_close(snapshot.rows[0], expected)
    torch.testing.assert_close(snapshot.first_response_logits, torch.arange(9).float()[None, None])
    assert all(bool((c == -999).all()) for c in caches)
    # Last request may never receive a finished callback; explicit drain exports it.
    exporter.begin(tmp_path/'v1', 1)
    runner.input_batch.num_computed_tokens_cpu[0] = 0
    runner.execute_model(schedule())
    reply = exporter.finish()
    assert reply['snapshots']['a']['version'] == 1
    trajectory.history_ref = reply['snapshots']['a']
    with pytest.raises(ValueError, match='stale'):
        load_rollout_snapshot(trajectory, NS(cfg=body.latent_cfg))


@pytest.mark.parametrize('field,value,match', [
    ('version', 7, 'stale'), ('request_id', 'wrong', 'stale'),
    ('length', 99, 'length'), ('token_ids', [1], 'token ids')])
def test_reject_bad_reference(tmp_path, field, value, match):
    runner, body, _ = fake_runner()
    exporter = RolloutCacheExporter(runner);exporter.begin(tmp_path/'v0', 0)
    runner.execute_model(schedule());ref = exporter.finish()['snapshots']['a']
    ref['token_ids'] = [4, 5, 6];ref[field] = value
    t = Trajectory(torch.tensor([[4, 5, 6]]), 2, 0, history_ref=ref, request_id='a')
    with pytest.raises(ValueError, match=match):load_rollout_snapshot(t, NS(cfg=body.latent_cfg))


def test_reject_over_scheduled_export(tmp_path):
    """Async-scheduling shape: one extra executed token exports P+n rows for a P+n-1 trajectory."""
    runner, body, _ = fake_runner()
    exporter = RolloutCacheExporter(runner);exporter.begin(tmp_path/'v0', 0)
    runner.execute_model(schedule())
    runner.input_batch.num_computed_tokens_cpu[0] = 2
    runner.execute_model(schedule(1))
    ref = exporter.finish()['snapshots']['a']
    assert ref['length'] == 3
    ref['token_ids'] = [4, 5, 6]
    t = Trajectory(torch.tensor([[4, 5, 6]]), 2, 0, history_ref=ref, request_id='a')
    with pytest.raises(ValueError, match='exported 3.*needs 2'):
        load_rollout_snapshot(t, NS(cfg=body.latent_cfg))


def test_rollout_replay_never_collects(tmp_path):
    from ouro_depth.tests.test_khop_replay import tiny_double
    from ouro_depth.latent.history_snapshot import collect_snapshot
    from ouro_depth.latent.khop_replay import replay_batch_khop
    model, student, ids = tiny_double()
    ids = ids[:, :7];prompt = 4
    snapshot = collect_snapshot(model, student, ids, prompt, serving_numerics=True)
    path = tmp_path/'cache.pt'
    torch.save(dict(schema='s6-rollout-cache-v1', version=2, request_id='x', cfg=student.cfg,
        prompt_ids=ids[0,:prompt].tolist(), length=6, rows=snapshot.rows,
        first_response_logits=snapshot.first_response_logits), path)
    ref = dict(path=str(path), version=2, request_id='x', length=6, token_ids=ids[0].tolist())
    t = Trajectory(ids,prompt,2,torch.zeros(1,3),False,ref,'x')
    def loss(lp, old, teacher, mask, normalizer):return (lp*mask).sum()/normalizer
    with patch('ouro_depth.latent.khop_replay.collect_snapshot', side_effect=AssertionError('serial collect')):
        metrics = replay_batch_khop(model,student,t,hops=3,normalizer=3,teacher_logp=torch.zeros(1,3),
            opd_loss=loss,serving_numerics=True,history_source='rollout')
    assert metrics['history_collect_seconds'] == 0
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in student.parameters())


def test_v2_scheduler_blocks_and_sampling_logits(tmp_path):
    runner, body, caches = fake_runner()
    del runner._update_states
    runner.block_tables = NS(blocks_per_kv_block=[2, 1], kernel_block_sizes=[2, 2])
    runner.req_states = NS(req_id_to_index={'a': 0}, num_computed_tokens_np=[0])
    runner.finish_requests = lambda s: [c.fill_(-99) for c in caches] if s.finished_req_ids else None
    runner.update_requests = lambda s: None
    model = NS(model=body, compute_logits=lambda h: torch.arange(9).float()[None])
    runner.get_model = lambda: model
    runner.sample = lambda h, batch, grammar: model.compute_logits(h)
    exporter = RolloutCacheExporter(runner);exporter.begin(tmp_path/'v2', 0)
    s = schedule();s.preempted_req_ids = set()
    s.scheduled_new_reqs = [NS(req_id='a', num_computed_tokens=0, prompt_token_ids=[4,5],block_ids=([0],[1]))]
    s.scheduled_cached_reqs = NS(req_ids=[],new_block_ids=[])
    runner.update_requests(s)
    assert exporter.pending['a']['blocks'][0][0].tolist() == [0,1]
    runner.sample(None,NS(req_ids=['a']),None)
    runner.req_states.num_computed_tokens_np[0] = 2
    s.num_scheduled_tokens = {'a':1};s.scheduled_new_reqs=[]
    s.scheduled_cached_reqs=NS(req_ids=['a'],new_block_ids=[([], [2])])
    runner.update_requests(s)
    expected=torch.cat([caches[0][[0,0,1],0,[0,1,0]], caches[1][[1,1,2],0,[0,1,0]]],-1)[None]
    s.finished_req_ids={'a'};runner.finish_requests(s)
    ref=exporter.finish()['snapshots']['a'];ref['token_ids']=[4,5,6,7]
    snap=load_rollout_snapshot(Trajectory(torch.tensor([[4,5,6,7]]),2,0,history_ref=ref,request_id='a'),NS(cfg=body.latent_cfg))
    torch.testing.assert_close(snap.rows[0],expected)


def test_external_internal_request_id_mapping():
    from ouro_depth.vllm_latent.rollout_worker import generate_with_request_ids
    def assign(req):
        req.external_req_id = req.request_id
        req.request_id += '-random-suffix'
    processor=NS(assign_request_id=assign)
    def generate(prompts, params, use_tqdm):
        for request_id in ('a','b'):processor.assign_request_id(NS(request_id=request_id))
        return [NS(request_id='b'),NS(request_id='a')]
    llm=NS(llm_engine=NS(input_processor=processor),generate=generate)
    outputs,mapping=generate_with_request_ids(llm,[[1],[1]],None)
    assert [mapping[o.request_id] for o in outputs] == ['b-random-suffix','a-random-suffix']
    assert processor.assign_request_id is assign
