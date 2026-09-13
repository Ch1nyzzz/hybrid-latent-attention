"""Import one pinned official checkpoint; never execute its modeling code."""
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download


ROOT = Path('/data/erv1n/ouro-depth-20260913')
REPO = 'tomg-group-umd/huginn-0125'
REVISION = 'bb6621b65e90b6a4b9b29ef88dc83866d450470c'
DESTINATION = ROOT / 'huginn_model'
STATUS = ROOT / 'diagnostics/huginn-engineering/import-status.json'


def record(value):
    value['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(value), flush=True)


def main():
    if DESTINATION.exists():
        raise FileExistsError('Inspect the existing import before any recovery')
    state = {'phase': 'initializing', 'pid': os.getpid(), 'repo': REPO,
             'revision': REVISION, 'destination': str(DESTINATION),
             'model_code_executed': False, 'gpu_used': False}
    with STATUS.open('x') as handle:
        json.dump(state, handle)
    try:
        info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise ValueError('Official API revision differs from the pinned checkpoint')
        selected = [item for item in info.siblings if
                    item.rfilename.endswith(('.json', '.py', '.safetensors'))
                    or item.rfilename == 'README.md']
        expected = {item.rfilename: {'size': item.size, 'git_blob': item.blob_id,
                    'sha256': item.lfs.sha256 if item.lfs else None} for item in selected}
        state.update(phase='downloading', files=expected)
        record(state)
        start = time.monotonic()
        snapshot_download(REPO, revision=REVISION, local_dir=DESTINATION,
                          cache_dir=ROOT / 'hf_cache/hub',
                          allow_patterns=list(expected), max_workers=4)
        state.update(phase='verifying_import', download_seconds=time.monotonic()-start)
        record(state)
        verified = {}
        for name, identity in expected.items():
            path = DESTINATION / name
            if path.stat().st_size != identity['size']:
                raise ValueError(f'Imported size mismatch: {name}')
            algorithm = 'sha256' if identity['sha256'] else 'git_blob_sha1'
            digest = hashlib.sha256() if identity['sha256'] else hashlib.sha1()
            if not identity['sha256']:
                digest.update(f'blob {identity["size"]}\0'.encode())
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b''):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != (identity['sha256'] or identity['git_blob']):
                raise ValueError(f'Imported digest mismatch: {name}')
            verified[name] = {'size': identity['size'], 'algorithm': algorithm, 'digest': actual}
        state.update(phase='completed', elapsed_seconds=time.monotonic()-start,
                     verified_files=verified)
        record(state)
        (ROOT / 'artifacts/huginn-model-source.json').write_text(json.dumps(state, indent=2)+'\n')
    except BaseException as error:
        state.update(phase='failed', error=repr(error),
                     note='Inspect this PID and partial files before recovery; no model was executed')
        record(state)
        raise


if __name__ == '__main__':
    main()
