"""Verify and restore the existing packed corpus once per Trisol pod."""
import hashlib,json,tarfile
from pathlib import Path
source=Path('/trisol/input/datasets/ds-0');target=Path('/work/expanded-corpus')
target.mkdir(parents=True,exist_ok=True)
spec=json.loads((source/'transfer.json').read_text())
archive=Path('/work/corpus.tar.gz')
with archive.open('wb') as output:
    for part in spec['parts']:
        assert Path(part['name']).name==part['name']
        content=(source/part['name']).read_bytes()
        assert len(content)==part['bytes']
        assert hashlib.sha256(content).hexdigest()==part['sha256']
        output.write(content)
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    assert {m.name for m in members}==set(spec['files'])
    assert all(m.isfile() and Path(m.name).name==m.name for m in members)
    bundle.extractall(target)
assert json.loads((target/'audit.json').read_text())['complete']
print(json.dumps(dict(event='corpus_restored',files=len(spec['files']),manifest_sha256=hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest())),flush=True)
