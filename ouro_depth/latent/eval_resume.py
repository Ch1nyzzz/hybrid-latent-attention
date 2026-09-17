"""Validated completed-answer imports when changing evaluation scheduling."""
import json
import math
from pathlib import Path


def load_completed(directory, samples, protocol):
    if not directory:
        return [], {}
    root = Path(directory)
    metadata = json.loads((root / 'resume-protocol.json').read_text())
    elapsed = metadata.get('elapsed_seconds', 0)
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid resumed elapsed_seconds')
    for key, value in protocol.items():
        if metadata.get(key) != value:
            raise ValueError(f'Resume protocol mismatch: {key}')
    expected = {(row['id'], k): row['answer'] for row, k in samples}
    rows, seen = [], set()
    for line in (root / f"shard{protocol['shard']}.jsonl").read_text().splitlines():
        row = json.loads(line)
        pair = (row['id'], row['sample'])
        if pair in seen or pair not in expected or row['gold'] != expected[pair]:
            raise ValueError(f'Duplicate or foreign resumed sample: {pair}')
        if type(row['correct']) is not bool or type(row['truncated']) is not bool:
            raise ValueError('Invalid resumed score flags')
        if type(row['tokens']) is not int or not 0 < row['tokens'] <= protocol['max_new'] or not isinstance(row['text'], str):
            raise ValueError('Invalid resumed answer')
        seen.add(pair); rows.append(row)
    return rows, metadata
