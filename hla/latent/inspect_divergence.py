import json
from pathlib import Path

def main():
    fv_dir = Path.home() / "diagnostic_run" / "output" / "eval_full_v_20260921_152946"
    lat_dir = Path.home() / "diagnostic_run" / "output" / "eval_latent_20260921_154430"

    def load_probs(d):
        res = {}
        for sf in sorted(d.glob("shard*.jsonl")):
            for l in open(sf):
                if l.strip():
                    item = json.loads(l)
                    res[item["id"]] = item
        return res

    fv = load_probs(fv_dir)
    lat = load_probs(lat_dir)

    print("=" * 80)
    print("=== GROUP 1: Full-V OK, Latent FAIL ===")
    print("=" * 80)
    for p in ["math500-001", "math500-012", "math500-014", "math500-028"]:
        f = fv[p]
        l = lat[p]
        print(f"ID: {p} | Gold: {f['gold']}")
        print(f"  [Full-V]  Correct: {f['correct']} | Tokens: {f['tokens']:>4} | Trunc: {f['truncated']} | Pred: {f['pred']}")
        print(f"  [Latent]  Correct: {l['correct']} | Tokens: {l['tokens']:>4} | Trunc: {l['truncated']} | Pred: {l['pred']}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("=== GROUP 2: Latent OK, Full-V FAIL ===")
    print("=" * 80)
    for p in ["math500-015", "math500-024", "math500-032", "math500-034"]:
        f = fv[p]
        l = lat[p]
        print(f"ID: {p} | Gold: {f['gold']}")
        print(f"  [Full-V]  Correct: {f['correct']} | Tokens: {f['tokens']:>4} | Trunc: {f['truncated']} | Pred: {f['pred']}")
        print(f"  [Latent]  Correct: {l['correct']} | Tokens: {l['tokens']:>4} | Trunc: {l['truncated']} | Pred: {l['pred']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
