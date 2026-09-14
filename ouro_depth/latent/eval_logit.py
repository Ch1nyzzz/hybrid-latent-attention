"""End-to-end evaluation of a stage-1 student: swap the latent cache into all layers and measure final-logit KL
against the teacher on held-out blocks, for history exit depth tau = 1..T (None = no early exit, same as T).

python -m ouro_depth.latent.eval_logit --model-path M --data-dir D --student S.pt --output O [--blocks 64]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from .register import LatentStudent
from .swap import logit_kl
from .vendor_model import load_teacher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--student", required=True)
    p.add_argument("--output", required=True); p.add_argument("--blocks", type=int, default=64); p.add_argument("--micro-batch", type=int, default=2)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.student, map_location="cpu")
    cfg = ck["cfg"]
    model = load_teacher(args.model_path, cfg["loops"], device)
    student = LatentStudent(**cfg).to(device).eval(); student.load_state_dict(ck["student"])
    dev = np.load(Path(args.data_dir) / "dev.npy")[: args.blocks]
    exits = list(range(1, cfg["loops"])) + [None]
    acc = {str(e): {"kl": 0.0, "top1_agree": 0.0, "nll_teacher": 0.0, "nll_swapped": 0.0} for e in exits}; n = 0
    for i in range(0, len(dev), args.micro_batch):
        ids = torch.from_numpy(dev[i:i + args.micro_batch].astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            r = logit_kl(model, student, ids, exits)
        for k, v in r.items():
            for m in v: acc[k][m] += v[m]
        n += 1
        print("batch", i, {k: round(v["kl"] / n, 4) for k, v in acc.items()}, flush=True)
    res = {k: {m: v[m] / n for m in v} for k, v in acc.items()}
    res["rows"] = "history exit loop tau (None = full depth); logits taken at the final loop"; res["student"] = args.student; res["cfg"] = cfg; res["step"] = ck.get("step")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "logit_kl.json", "w"), indent=1)
    print(json.dumps({"LOGIT_KL": {k: {m: round(x, 4) for m, x in v.items()} for k, v in res.items() if isinstance(v, dict) and "kl" in v}}), flush=True)
    print("LOGIT_DONE", flush=True)


if __name__ == "__main__":
    main()
