#!/usr/bin/env bash
set -euo pipefail
cd /data/erv1n/ouro-depth-20260913
export HF_HOME=/data/erv1n/ouro-depth-20260913/hf_cache
export PIP_CACHE_DIR=/data/erv1n/ouro-depth-20260913/pip_cache
export HF_HUB_DISABLE_XET=1
if [ ! -x .venv/bin/python ]; then
  /usr/bin/python3.12 -m venv .venv
fi
printf '%s\n' /data/erv1n/train_venv/lib/python3.12/site-packages > .venv/lib/python3.12/site-packages/existing-runtime.pth
.venv/bin/python -m pip install 'transformers==4.56.2' 'huggingface-hub==0.36.2' 'tokenizers>=0.22,<0.23' pytest
.venv/bin/python - <<'PY'
import os,json,pathlib
from huggingface_hub import HfApi,snapshot_download
import torch,transformers
print('RUNTIME',torch.__version__,transformers.__version__,flush=True)
repo='ByteDance/Ouro-1.4B'
info=HfApi().model_info(repo)
path=snapshot_download(repo,revision=info.sha,allow_patterns=['*.json','*.txt','*.py','*.safetensors'],local_dir='base_model')
pathlib.Path('artifacts').mkdir(exist_ok=True)
pathlib.Path('artifacts/model_source.json').write_text(json.dumps({'repository':repo,'revision':info.sha,'path':str(path),'torch':torch.__version__,'transformers':transformers.__version__},indent=2)+'\n')
print('MODEL_READY',info.sha,path,flush=True)
PY
