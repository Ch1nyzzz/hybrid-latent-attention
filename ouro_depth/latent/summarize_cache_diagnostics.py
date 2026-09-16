"""Summarize the completed fixed-prefix S5 diagnostic without GPU execution."""
import argparse
import json
import math
from pathlib import Path


def mean_metrics(rows):
    count = sum(r["positions"] for r in rows)
    for row in rows:
        values = row["kl_by_position"]
        assert len(values) == row["positions"]
        assert all(math.isfinite(v) for v in values)
        assert abs(sum(values) / len(values) - row["kl"]) < 1e-6
        assert abs(row["top1_agree"] * len(values) - round(row["top1_agree"] * len(values))) < 1e-4
    fields = ("kl", "top1_agree", "reference_top1_logprob_abs_diff", "eos_probability_abs_diff")
    return {"positions": count, **{k: sum(r[k] * r["positions"] for r in rows) / count for k in fields}}


def summarize(folder):
    def read(name):
        return json.loads((folder / f"{name}.json").read_text())

    histories = [read(f"history_dev{i}-p128-s128") for i in range(4)]
    history = {k: mean_metrics([r[k] for r in histories]) for k in histories[0] if k != "case"}
    assert history["teacher_vs_exact"]["positions"] == 512
    parity = read("vllm_parity")
    expected = {f"dev{i}-p128-s128": 128 for i in range(4)} | {"dev4-p1024-s32": 32, "packed-dev-p4096-s16": 16}
    assert len(parity) == 12
    serving = {}
    for after in (False, True):
        rows = [r for r in parity if r["finalize_after_read"] is after]
        assert len(rows) == 6 and {r["case"] for r in rows} == set(expected)
        for row in rows:
            assert row["prefill"]["positions"] == 1
            assert row["decode"]["positions"] == expected[row["case"]]
            assert row["all"]["positions"] == expected[row["case"]] + 1
        serving["after_read" if after else "before_read"] = {
            **{part: mean_metrics([r[part] for r in rows]) for part in ("all", "prefill", "decode")},
            "short_decode": mean_metrics([r["decode"] for r in rows if expected[r["case"]] == 128]),
        }
    residual = read("finalizer_residual")
    gradients = read("history_gradients")
    assert len(residual) == 24 and len(gradients) == 2
    assert gradients[0]["detached_between_chunks"] and not gradients[1]["detached_between_chunks"]
    assert all(v is None for v in gradients[0]["history_gradient_norms"])
    assert all(v is not None and v > 0 for v in gradients[1]["history_gradient_norms"])
    assert gradients[1]["forward_max_abs_diff_vs_detached"] == 0
    manifest = read("diagnostic_manifest")
    return {
        "job_id": "2099920959799570432",
        "manifest": manifest,
        "student_cfg": read("checkpoint")["cfg"],
        "history": history,
        "serving": serving,
        "finalizer": {"mean_over_calls": sum(r["mean"] * r["calls"] for r in residual) / sum(r["calls"] for r in residual),
                      "max_observed_relative_residual": max(r["max"] for r in residual), "by_layer": residual},
        "gradients": gradients,
        "derived": {
            "real_vs_proxy_teacher_kl_relative_increase": history["teacher_vs_exact"]["kl"] / history["teacher_vs_two_pass"]["kl"] - 1,
            "chunk16_vs_proxy_real_reference_kl_reduction": 1 - history["exact_vs_chunk16"]["kl"] / history["exact_vs_two_pass"]["kl"],
            "serving_decode_kl_reduction": 1 - serving["after_read"]["decode"]["kl"] / serving["before_read"]["decode"]["kl"],
        },
        # Per-case and per-position records remain in the local input artifacts.
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("history", "serving", "derived")}, indent=2))


if __name__ == "__main__":
    main()
