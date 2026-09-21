"""Export S6 paged caches before vLLM retires/reuses request blocks.

Hooks run outside the captured model forward. This adapter deliberately supports
only non-speculative, single-worker TRITON_ATTN generation without preemption.
"""
from pathlib import Path
import time
import torch

SCHEMA = 's6-rollout-cache-v1'


class RolloutCacheExporter:
    def __init__(self, runner):
        if getattr(runner, 'speculative_config', None) is not None:
            raise ValueError('Cache export does not support speculative decoding')
        self.runner = runner
        model = runner.get_model()
        while not hasattr(model, 'model') and hasattr(model, 'unwrap'):
            model = model.unwrap()
        self.body = model.model
        self.layer_groups = {name: i for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                             for name in group.layer_names}
        self.active = False
        self.pending = {}
        self.finished = {}
        if not hasattr(runner, '_update_states'):
            self.install_v2_hooks()
            return
        update, execute = runner._update_states, runner.execute_model

        def update_states(schedule):
            if self.active:
                for req_id in schedule.finished_req_ids:
                    self.export(req_id)
                if getattr(schedule, 'preempted_req_ids', None):
                    raise RuntimeError('Preempted S6 history cannot be exported')
            result = update(schedule)
            if self.active:
                self.observe_inputs(schedule)
            return result

        def execute_model(schedule, *args, **kwargs):
            result = execute(schedule, *args, **kwargs)
            if self.active and schedule.total_num_scheduled_tokens:
                state = runner.execute_model_state
                if state is None or state.logits.shape[0] != len(runner.input_batch.req_ids):
                    raise RuntimeError('Unsupported vLLM logits/request layout')
                for req_id, index in runner.input_batch.req_id_to_index.items():
                    item = self.pending.get(req_id)
                    if item is not None and item['first_logits'] is None:
                        if item['length'] != len(item['prompt']):
                            raise RuntimeError('Export requires a complete unchunked prompt prefill')
                        item['first_logits'] = state.logits[index:index+1].detach().cpu().clone().unsqueeze(1)
            return result

        runner._update_states = update_states
        runner.execute_model = execute_model

    def install_v2_hooks(self):
        runner = self.runner
        finish, update, sample = runner.finish_requests, runner.update_requests, runner.sample
        compute_logits = runner.get_model().compute_logits
        self.sample_batch = None

        def finish_requests(schedule):
            if self.active:
                if schedule.preempted_req_ids:
                    raise RuntimeError('Preempted S6 history cannot be exported')
                for req_id in schedule.finished_req_ids:
                    self.export(req_id)
            return finish(schedule)

        def update_requests(schedule):
            result = update(schedule)
            if self.active:
                self.observe_v2_inputs(schedule)
            return result

        def sample_tokens(hidden_states, input_batch, grammar_output):
            self.sample_batch = input_batch
            try:
                return sample(hidden_states, input_batch, grammar_output)
            finally:
                self.sample_batch = None

        def logits(hidden_states, *args, **kwargs):
            output = compute_logits(hidden_states, *args, **kwargs)
            if self.active and self.sample_batch is not None:
                req_ids = self.sample_batch.req_ids
                if output.shape[0] != len(req_ids):
                    raise RuntimeError('Unsupported vLLM logits/request layout')
                for index, req_id in enumerate(req_ids):
                    item = self.pending[req_id]
                    if item['first_logits'] is None:
                        if item['length'] != len(item['prompt']):
                            raise RuntimeError('Export requires unchunked prompt prefill')
                        item['first_logits'] = output[index:index+1].detach().cpu().clone().unsqueeze(1)
            return output

        runner.finish_requests = finish_requests
        runner.update_requests = update_requests
        runner.sample = sample_tokens
        runner.get_model().compute_logits = logits

    def observe_v2_inputs(self, schedule):
        runner = self.runner
        for request in schedule.scheduled_new_reqs:
            if request.req_id in self.pending or request.num_computed_tokens != 0:
                raise RuntimeError('Export requires fresh non-prefix-cached requests')
            self.pending[request.req_id] = dict(length=0, prompt=list(request.prompt_token_ids),
                first_logits=None, scheduler_blocks=[list(x) for x in request.block_ids])
        cached = schedule.scheduled_cached_reqs
        for req_id, new_blocks in zip(cached.req_ids, cached.new_block_ids):
            if new_blocks is not None:
                for blocks, added in zip(self.pending[req_id]['scheduler_blocks'], new_blocks):
                    blocks.extend(added)
        for req_id, count in schedule.num_scheduled_tokens.items():
            item = self.pending[req_id]
            index = runner.req_states.req_id_to_index[req_id]
            start = int(runner.req_states.num_computed_tokens_np[index])
            if item['length'] != start:
                raise RuntimeError('Discontinuous/recomputed rollout history')
            item['length'] = start+count
            item['blocks'] = []
            for group, blocks in enumerate(item['scheduler_blocks']):
                ratio = runner.block_tables.blocks_per_kv_block[group]
                physical = [b*ratio+k for b in blocks for k in range(ratio)]
                item['blocks'].append((torch.tensor(physical, dtype=torch.long),
                                       runner.block_tables.kernel_block_sizes[group]))

    def begin(self, directory, version):
        if self.active or self.pending:
            raise RuntimeError('Previous cache export has not drained')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.version = version
        self.finished = {}
        self.seconds = 0.
        self.active = True

    def observe_inputs(self, schedule):
        runner = self.runner
        for req_id, count in schedule.num_scheduled_tokens.items():
            index = runner.input_batch.req_id_to_index[req_id]
            request = runner.requests[req_id]
            start = int(runner.input_batch.num_computed_tokens_cpu[index])
            old = self.pending.get(req_id)
            if old is not None and old['length'] != start:
                raise RuntimeError('Discontinuous/recomputed rollout history')
            blocks = []
            for table in runner.input_batch.block_table:
                n = int(table.num_blocks_per_row[index])
                blocks.append((table.get_cpu_tensor()[index, :n].clone(), table.block_size))
            self.pending[req_id] = dict(length=start+count, blocks=blocks,
                prompt=list(request.prompt_token_ids),
                first_logits=old['first_logits'] if old is not None else None)

    @torch.no_grad()
    def export(self, req_id):
        item = self.pending.pop(req_id, None)
        if item is None:
            return
        tick = time.perf_counter()
        if item['first_logits'] is None:
            raise RuntimeError('Missing prompt logits at request retirement')
        rows = []
        for layer in self.body.layers:
            parts = []
            for attn in (layer.self_attn.attn_main, layer.self_attn.attn_l1):
                cache = attn.kv_cache
                blocks, block_size = item['blocks'][self.layer_groups[attn.layer_name]]
                if cache.ndim != 4 or cache.shape[1] != 1 or cache.shape[2] != block_size:
                    raise RuntimeError('Unsupported TRITON_ATTN cache layout')
                pos = torch.arange(item['length'], device=cache.device)
                physical = blocks.to(device=cache.device, dtype=torch.long)[pos // block_size]
                parts.append(cache[physical, 0, pos % block_size, :])
            rows.append(torch.cat(parts, -1).unsqueeze(0).cpu())
        path = self.directory / f'{len(self.finished)}.pt'
        payload = dict(schema=SCHEMA, version=self.version, request_id=req_id,
            cfg=self.body.latent_cfg, prompt_ids=item['prompt'], length=item['length'],
            first_response_logits=item['first_logits'], rows=tuple(rows))
        temp = path.with_suffix('.tmp')
        torch.save(payload, temp)
        temp.replace(path)
        self.finished[req_id] = dict(path=str(path), request_id=req_id, version=self.version,
                                    length=item['length'], bytes=path.stat().st_size)
        self.seconds += time.perf_counter()-tick

    def finish(self):
        if not self.active:
            raise RuntimeError('No cache export in progress')
        for req_id in list(self.pending):
            self.export(req_id)
        self.active = False
        return dict(version=self.version, snapshots=self.finished, export_seconds=self.seconds)
