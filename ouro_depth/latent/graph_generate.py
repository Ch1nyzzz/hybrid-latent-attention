"""Inference-only CUDA graphs over the qualified S6 batched reference engine.

Padding remains masked. A physical cursor addresses packed history independently
of each row's logical RoPE position. Training never imports this module.
"""
from __future__ import annotations

import torch
from .generate import BatchedLatentDecoder


class GraphRollingStep:
    """Keep mutable decode state at fixed addresses for one capacity bucket."""
    def __init__(self, engine, minimum_capacity=256, compact_finished=False):
        if engine.positions.device.type != 'cuda':
            raise ValueError('CUDA graph decode requires CUDA')
        self.engine = engine
        self.used = engine.prefix[0].shape[1]
        self.minimum_capacity = minimum_capacity
        self.capacity = 0
        self.graph = None
        self.capture_count = 0
        self.compact_finished = compact_finished
        self.original_batch = engine.positions.shape[0]
        self.row_ids = torch.arange(self.original_batch, device=engine.positions.device)
        self.full_output = None

    @property
    def batch_size(self):
        return self.row_ids.numel()

    @property
    def positions(self):
        return self.engine.positions

    @property
    def prefix_mask(self):
        return self.engine.prefix_mask[:, :self.used]

    def _run(self):
        e = self.engine
        e.clear_live()
        e.positions = self.static_positions
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            logits, _ = e.step(self.ids, self.valid)
        for dst, src in zip(e.prefix, e.last_written):
            dst.index_copy_(1, self.cursor, src)
        e.prefix_mask.index_copy_(1, self.cursor, self.valid)
        self.static_positions.copy_(e.positions)
        self.cursor.add_(1)
        e.positions = self.static_positions
        e.clear_live()
        return logits

    @torch.no_grad()
    def _capture(self, ids, valid, keep=None):
        e = self.engine
        # Release the old graph before allocating a new pool. Cache contents are
        # external graph inputs, not graph-private allocations.
        self.graph = None
        self.output = None
        capacity = (self.used // self.minimum_capacity + 1) * self.minimum_capacity
        old_prefix, old_mask = list(e.prefix), e.prefix_mask
        positions = e.positions.clone() if keep is None else e.positions.index_select(0, keep)
        # Migrate one layer at a time, releasing each old allocation immediately.
        # This bounds growth to one extra layer instead of two full caches.
        e.prefix = ()
        e.storage = None
        new_prefix = []
        for index in range(len(old_prefix)):
            src = old_prefix[index]
            dst = src.new_zeros(ids.shape[0], capacity, src.shape[-1])
            if keep is None:
                dst[:, :self.used].copy_(src[:, :self.used])
            else:
                dst[:, :self.used].copy_(src[:, :self.used].index_select(0, keep))
            new_prefix.append(dst)
            old_prefix[index] = None
            del src, dst
        e.prefix = tuple(new_prefix)
        del old_prefix, new_prefix
        e.prefix_mask = old_mask.new_zeros(ids.shape[0], capacity)
        selected_mask = old_mask if keep is None else old_mask.index_select(0, keep)
        e.prefix_mask[:, :self.used].copy_(selected_mask[:, :self.used])
        del old_mask, selected_mask
        self.static_positions = positions.clone()
        self.cursor = torch.tensor([self.used], device=ids.device, dtype=torch.long)
        self.ids, self.valid = ids.clone(), valid.clone()
        self.capacity = capacity

        def restore():
            # Warmup/capture append exactly one column; old history is read-only.
            for row in e.prefix:
                row[:, self.used].zero_()
            e.prefix_mask[:, self.used].zero_()
            self.static_positions.copy_(positions)
            self.cursor.fill_(self.used)
            e.positions = self.static_positions
            e.clear_live()

        stream = torch.cuda.Stream(device=ids.device)
        stream.wait_stream(torch.cuda.current_stream(ids.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._run()
                restore()
        torch.cuda.current_stream(ids.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.output = self._run()
        torch.cuda.current_stream(ids.device).wait_stream(stream)
        restore()
        self.graph = graph
        self.capture_count += 1

    @torch.no_grad()
    def step(self, ids, valid=None):
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        keep = None
        # Retain a minimum compute batch of eight: very small BF16 GEMMs can
        # change accumulation algorithms enough to fail the numerical gate.
        if self.compact_finished and self.batch_size > 8 and (self.graph is None or self.used % 32 == 0):
            live = valid.index_select(0, self.row_ids)[:, 0]
            count = int(live.sum())
            if 0 < count <= self.batch_size // 2:
                keep = live.nonzero(as_tuple=True)[0]
                if count < 8:
                    padding = (~live).nonzero(as_tuple=True)[0][:8-count]
                    keep = torch.cat((keep, padding)).sort().values
                self.row_ids = self.row_ids.index_select(0, keep)
        selected_ids = ids.index_select(0, self.row_ids)
        selected_valid = valid.index_select(0, self.row_ids)
        if self.graph is None or self.used >= self.capacity or keep is not None:
            self._capture(selected_ids, selected_valid, keep)
        self.ids.copy_(selected_ids)
        self.valid.copy_(selected_valid)
        self.graph.replay()
        self.used += 1
        if self.batch_size == self.original_batch:
            return self.output, None
        if self.full_output is None:
            self.full_output = self.output.new_zeros(self.original_batch, *self.output.shape[1:])
        self.full_output.index_copy_(0, self.row_ids, self.output)
        return self.full_output, None

    def detach_history(self):
        # Cache appends are part of the graph, with no autograd state.
        pass


class GraphLatentDecoder(BatchedLatentDecoder):
    def __init__(self, *args, compact_finished=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_finished = compact_finished

    @torch.no_grad()
    def prefill_batch(self, prompts):
        engine, logits = super().prefill_batch(prompts)
        return GraphRollingStep(engine, compact_finished=self.compact_finished), logits
