"""Unpadded S6 generation interface over the shared chunk engine."""
from .batched_engine import BatchedRollingEngine


class RollingEngine(BatchedRollingEngine):
    def __init__(self, model, student, checkpointing=True, prompt_chunk_size=256):
        super().__init__(model, student, checkpointing)
        self.prompt_chunk_size = prompt_chunk_size

    def prefill(self, ids, targets=None):
        return super().prefill(ids, targets=targets, chunk_size=self.prompt_chunk_size or ids.shape[1])
