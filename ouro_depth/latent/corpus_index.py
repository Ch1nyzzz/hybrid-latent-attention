"""Seekable corpus and deterministic, source-stratified epoch sampling."""
import hashlib
import json
from pathlib import Path
import random


class RecordIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {"openr1": [], "fineweb": []}
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                self.rows[row["source"]].append((offset, len(row["input_ids"])))
        self.stream = self.path.open("rb")
        self._eligible = {}
        self._orders = {}

    def close(self):
        self.stream.close()

    def _read(self, offset):
        self.stream.seek(offset)
        return json.loads(self.stream.readline())

    def sample(self, source, rng, min_length=2):
        # Preserve the historical draw sequence for old checkpoints/runs.
        candidates = self.rows[source]
        if not any(length >= min_length for _, length in candidates):
            raise ValueError(f"No eligible {source} record in {self.path}")
        while True:
            offset, length = candidates[rng.randrange(len(candidates))]
            if length >= min_length:
                return self._read(offset)

    def sample_at(self, sequence_id, *, seed, stage, min_length=64):
        """60/40 example mix; each source visits all eligible chunks before reuse.

        sequence_id is stage-local and global across ranks, not rank-local.
        Selection is stateless: resume needs only the completed-step cursor,
        stage, seed and immutable corpus, and rank partitioning cannot change it.
        """
        if sequence_id < 0:
            raise ValueError("sequence_id must be nonnegative")
        cycle, slot = divmod(sequence_id, 5)
        source = "openr1" if slot < 3 else "fineweb"
        ordinal = cycle * (3 if source == "openr1" else 2) + (slot if slot < 3 else slot - 3)
        key = (source, min_length)
        if key not in self._eligible:
            self._eligible[key] = [offset for offset, length in self.rows[source] if length >= min_length]
        offsets = self._eligible[key]
        if not offsets:
            raise ValueError(f"No eligible {source} record in {self.path}")
        epoch, position = divmod(ordinal, len(offsets))
        order_key = (source, min_length, seed, str(stage), epoch)
        if order_key not in self._orders:
            entropy = hashlib.sha256(json.dumps(order_key).encode()).digest()
            order = list(range(len(offsets)))
            random.Random(int.from_bytes(entropy, "big")).shuffle(order)
            # Retain only the latest permutation for each source/selection rule.
            self._orders = {k: v for k, v in self._orders.items() if k[:2] != key}
            self._orders[order_key] = order
        return self._read(offsets[self._orders[order_key][position]])
