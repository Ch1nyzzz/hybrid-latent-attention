"""Frozen teacher targets shared across successive, fresh student updates.

Only fixed token sequences are accepted: no student activations or KV state is
cached. Exact-length buckets avoid changing the teacher's masking/positions.
"""
from collections import defaultdict
from dataclasses import dataclass

import torch

from .batched_recipe import ReplayBatch


@dataclass
class TeacherRecord:
    ids: torch.Tensor
    prompt: int
    logits: torch.Tensor
    targets: dict
    denominators: dict


class FrozenTeacherCache:
    @torch.no_grad()
    def __init__(self, examples, teacher, micro_batch=16):
        if not examples or micro_batch < 1:
            raise ValueError('Nonempty examples and positive teacher microbatch required')
        self.records = [None] * len(examples)
        self.forward_batches = []
        buckets = defaultdict(list)
        for index, (ids, prompt) in enumerate(examples):
            if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= prompt < ids.shape[1]:
                raise ValueError('Expected [1,L] IDs and a nonempty prompt/continuation')
            buckets[ids.shape[1]].append(index)
        topology = None
        for indices in buckets.values():
            for offset in range(0, len(indices), micro_batch):
                selected = indices[offset:offset + micro_batch]
                inputs = torch.cat([examples[i][0][:, :-1] for i in selected])
                logits, outputs = teacher(inputs)
                if logits.shape[:2] != inputs.shape or any(v.shape[:2] != inputs.shape for v in outputs.values()):
                    raise ValueError('Teacher target shape mismatch')
                if topology is None:
                    topology = set(outputs)
                elif set(outputs) != topology:
                    raise ValueError('Teacher target topology changed')
                self.forward_batches.append(len(selected))
                for j, index in enumerate(selected):
                    # Views retain the batched outputs. No student graph survives.
                    targets = {k: v[j:j+1].detach() for k, v in outputs.items()}
                    denominators = {k: v.float().square().mean().clamp_min(1e-8)
                                    for k, v in targets.items()}
                    ids, prompt = examples[index]
                    self.records[index] = TeacherRecord(ids.detach().clone(), prompt,
                                                        logits[j:j+1].detach(), targets, denominators)

    @property
    def bytes(self):
        """Logical cache payload, excluding allocator/workspace and ID overhead."""
        return sum(r.logits.numel() * r.logits.element_size() +
                   sum(v.numel() * v.element_size() for v in r.targets.values()) +
                   sum(v.numel() * v.element_size() for v in r.denominators.values())
                   for r in self.records)

    @torch.no_grad()
    def batch(self, indices, stage=2):
        if stage not in (1, 2, 3):
            raise ValueError('Invalid stage')
        rows = [self.records[i] for i in indices]
        if not rows:
            raise ValueError('Empty student batch')
        prompts = [r.ids.shape[1] - 1 if stage == 1 else r.prompt for r in rows]
        prompt = max(prompts)
        offsets = [prompt - n for n in prompts]
        length = max(off + r.ids.shape[1] - 1 for off, r in zip(offsets, rows))
        ids = rows[0].ids.new_zeros((len(rows), length))
        valid = torch.zeros_like(ids, dtype=torch.bool)
        logits = rows[0].logits.new_zeros((len(rows), length, rows[0].logits.shape[-1]))
        targets = {k: v.new_zeros((len(rows), length, v.shape[-1])) for k, v in rows[0].targets.items()}
        denoms = {k: torch.stack([r.denominators[k] for r in rows]) for k in targets}
        for i, (row, off) in enumerate(zip(rows, offsets)):
            n = row.ids.shape[1] - 1
            ids[i, off:off+n] = row.ids[0, :-1]
            valid[i, off:off+n] = True
            logits[i, off:off+n] = row.logits[0]
            for k, value in row.targets.items():
                targets[k][i, off:off+n] = value[0]
        return ReplayBatch(ids, valid, prompt, logits, targets, denoms)
