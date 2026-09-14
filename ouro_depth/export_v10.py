"""Export a V10 trainable checkpoint as a self-contained HF Ouro directory for vLLM (PROTOCOL-v10.md §5).

  python -m ouro_depth.export_v10 --model-path base_model --checkpoint runs/v10-short_t8-s20260921/checkpoint-1500 --T 8 --output exports/v10-short_t8-1500
The exported config carries total_ut_steps=T; vLLM (hf_overrides) may still override it at load time.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from .model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--T', type=int, required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} exists')
    model, tokenizer = load_model(args.model_path, device='cpu', dtype=torch.bfloat16, mode='full', checkpointing=False)
    model.load_trainable(args.checkpoint)
    base = model.base
    base.config.total_ut_steps = args.T
    base.config.use_cache = True
    base.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    for source in Path(args.model_path).glob('*.py'):
        shutil.copy(source, output / source.name)
    # The vendored config class drops auto_map; trust_remote_code loaders (vLLM on transformers 5.x) need it back.
    base_config = json.loads((Path(args.model_path) / 'config.json').read_text())
    config = json.loads((output / 'config.json').read_text())
    if 'auto_map' in base_config:
        config['auto_map'] = base_config['auto_map']
    config['total_ut_steps'] = args.T
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    weight = Path(args.checkpoint)
    weight = weight if weight.suffix == '.pt' else weight / 'trainable.pt'
    receipt = {'checkpoint': str(weight.resolve()), 'checkpoint_sha256': hashlib.sha256(weight.read_bytes()).hexdigest(),
               'total_ut_steps': args.T, 'exported_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (output / 'export.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
