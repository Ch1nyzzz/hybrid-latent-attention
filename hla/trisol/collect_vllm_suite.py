"""Collect a fused-vLLM suite run from its trisol job log into results/latent/<name>.json and print the peak table.

python hla/trisol/collect_vllm_suite.py --job 2100764456979005440 --out results/latent/lla-vllm-peak-20260918.json
python hla/trisol/collect_vllm_suite.py --log job.log --out ...        # from a saved log

The log is fetched with `trisol train logs <job> --history` unless --log is given; every `VLLM_SUITE_CASE {...}` /
`VLLM_SUITE_DONE {...}` line (any prefix, e.g. a pod tag) is parsed. The output has the same shape as the earlier
`s6-vllm-peak-20260917.json` (`job_id`, `done`, `cases`), and the peak table lists, per (prompt, method), the sweep
`concurrency: decode tok/s` pairs and the peak from the `peak-summary` rows.
"""
from __future__ import annotations

import argparse, json, re, subprocess, sys
from pathlib import Path

_ROW = re.compile(r"VLLM_SUITE_(CASE|DONE) (\{.*\})\s*$")


def parse_rows(text: str) -> tuple[list[dict], dict | None]:
    """(case rows in log order, the DONE summary or None); a line is taken once even if the log repeats it."""
    cases, done, seen = [], None, set()
    for line in text.splitlines():
        m = _ROW.search(line)
        if not m:
            continue
        payload = m.group(2)
        if payload in seen:
            continue
        seen.add(payload)
        obj = json.loads(payload)
        if m.group(1) == "DONE":
            done = obj
        else:
            cases.append(obj)
    return cases, done


def peak_table(cases: list[dict]) -> list[dict]:
    """One line per peak-summary row: prompt, method, kv pool, c_max, sweep pairs and the peak."""
    out = []
    for row in cases:
        if row.get("stage") != "peak-summary":
            continue
        sweep = [(t["concurrency"], t["decode_tok_per_s"], t.get("end_to_end_tok_per_s")) for t in row.get("sweep", [])]
        out.append({"prompt": row["prompt"], "method": row["method"], "kv_cache_tokens": row.get("kv_cache_tokens"), "c_max": row.get("c_max"),
                    "gen_tokens": row.get("gen_tokens"), "peak_decode_tok_per_s": row.get("peak_decode_tok_per_s"),
                    "peak_concurrency": row.get("peak_concurrency"), "sweep": sweep, "problems": row.get("problems", [])})
    return sorted(out, key=lambda r: (r["prompt"], r["method"]))


def format_table(table: list[dict]) -> str:
    lines = []
    for r in table:
        pairs = " ".join(f"{c}:{d if d is not None else '-'}" for c, d, _ in r["sweep"])
        lines.append(f"p{r['prompt']:<5} {r['method']:<7} pool={r['kv_cache_tokens']} c_max={r['c_max']} N={r['gen_tokens']} "
                     f"peak={r['peak_decode_tok_per_s']}@{r['peak_concurrency']}  sweep {pairs}" + (f"  PROBLEMS {r['problems']}" if r["problems"] else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job", default=""); p.add_argument("--log", default=""); p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if bool(args.job) == bool(args.log):
        p.error("give exactly one of --job or --log")
    text = Path(args.log).read_text(errors="replace") if args.log else subprocess.run(
        ["trisol", "train", "logs", args.job, "--history", "--tail", "200000"], capture_output=True, text=True, check=True).stdout
    cases, done = parse_rows(text)
    if not cases:
        print("no VLLM_SUITE_CASE rows found", file=sys.stderr)
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"job_id": args.job or None, "done": done, "cases": cases}, indent=1))
    print(f"{len(cases)} cases, done={done}", file=sys.stderr)
    print(format_table(peak_table(cases)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
