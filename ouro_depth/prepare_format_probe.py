"""Create paired development-only views differing solely by source indentation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

try:
    from .data import verify_row
    from .prepare_diagnostics import _write_json, _write_rows
except ImportError:
    from data import verify_row
    from prepare_diagnostics import _write_json, _write_rows


ORIGINAL = "diagnostic-format-original"
INDENTED = "diagnostic-format-indented"
EDGE = re.compile(r"[a-z]{2} -> [a-z]{2}")
INDENTED_EDGE = re.compile(r" ([a-z]{2} -> [a-z]{2})")


def indent_prompt(prompt: str) -> str:
    lines = prompt.split("\n")
    if sum(bool(EDGE.fullmatch(line)) for line in lines) != 25:
        raise ValueError("Expected exactly 25 unindented edge-source lines")
    return "\n".join(" " + line if EDGE.fullmatch(line) else line for line in lines)


def restore_prompt(prompt: str) -> str:
    lines = prompt.split("\n")
    if sum(bool(INDENTED_EDGE.fullmatch(line)) for line in lines) != 25:
        raise ValueError("Expected exactly 25 source lines with one introduced ASCII space")
    return "\n".join(line[1:] if INDENTED_EDGE.fullmatch(line) else line for line in lines)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_format_probe(root: str | Path) -> dict:
    root = Path(root).resolve()
    source = _rows(root / "data/diagnostic-onehop/dev.jsonl")
    original = _rows(root / "data" / ORIGINAL / "dev.jsonl")
    indented = _rows(root / "data" / INDENTED / "dev.jsonl")
    if source != original or len(original) != len(indented):
        raise ValueError("Original view must exactly preserve the source rows and paired sample count")
    if len({row["id"] for row in original}) != len(original):
        raise ValueError("The source development set contains duplicate query IDs")
    for first, second in zip(original, indented):
        solved = verify_row(first)
        if first["split"] != "dev" or first["family"] != "pointer_chasing" or first["difficulty"] != 1 or solved["context_size"] != 25:
            raise ValueError("Format probes require original 25-node, one-hop development examples")
        expected = copy.deepcopy(first)
        expected["prompt"] = indent_prompt(first["prompt"])
        if expected != second:
            raise ValueError("Indented view changed more than the prescribed source-line spaces")
        if len(second["prompt"]) - len(first["prompt"]) != 25:
            raise ValueError("Expected exactly 25 introduced ASCII spaces per prompt")
        restored = copy.deepcopy(second)
        restored["prompt"] = restore_prompt(second["prompt"])
        if restored != first:
            raise ValueError("Removing the introduced spaces did not reconstruct the original row")
        verify_row(restored)
    for name, variant in ((ORIGINAL, "original"), (INDENTED, "indented")):
        directory = root / "data" / name
        if (directory / "train.jsonl").exists() or (directory / "test.jsonl").exists() or (directory / "ood.jsonl").exists():
            raise ValueError("The format probe must contain development data only")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["count"] != len(source) or manifest["variant"] != variant or manifest["independent_new_examples"] != 0:
            raise ValueError("Incorrect paired-probe manifest")
        if manifest["train_split_present"] is not False or manifest["pair_key"] != "id":
            raise ValueError("Manifest must declare development-only pairing")
    return {"count": len(source), "paired_ids_identical": True, "answers_and_metadata_identical": True,
            "row_and_edge_order_identical": True, "introduced_ascii_spaces_per_prompt": 25,
            "original_prompts_exactly_reconstructed": True, "all_original_and_restored_rows_independently_solved": True,
            "train_split_present": False, "independent_new_examples": 0}


def tokenizer_audit_record() -> dict:
    """Persist measurements already obtained by the read-only remote CPU audit."""
    return {
        "schema_version": 1, "measurement_date_utc": "2026-09-13",
        "evidence_origin": {
            "host": "reds-lab", "runtime": "/data/erv1n/ouro-depth-20260913/.venv/bin/python",
            "tokenizer_path": "/data/erv1n/ouro-depth-20260913/base_model",
            "tokenizer_class_observed": "GPT2TokenizerFast",
            "source_rows": "/data/erv1n/ouro-depth-20260913/data/v1/train.jsonl",
            "selection": "First 256 pointer_chasing rows in stored training-file order; all training difficulties eligible",
            "method": "AutoTokenizer.from_pretrained(use_fast=True, local_files_only=True); return_offsets_mapping=True; select IDs whose character offsets overlap the queried label",
            "execution": "Read-only SSH CPU invocation; CUDA_VISIBLE_DEVICES empty; no model weights, logits, evaluation, or training loaded",
            "persistence": "This file records previously observed tool output; preparing format probes does not rerun the tokenizer audit",
        },
        "original_256_rows": {
            "n": 256, "query_source_same_token_ids": 2, "query_source_different_token_ids": 254,
            "query_source_same_token_count": 192, "query_source_different_token_count": 64,
            "query_label_token_count": {"1": 140, "2": 116},
            "source_label_token_count": {"1": 202, "2": 54},
            "prompt_tokens": {"min": 173, "mean": 185.6953125, "max": 200},
            "examples": [
                {"label": "hm", "query_ids": [294, 93], "query_tokens": ["Ġh", "m"], "source_ids": [28150], "source_tokens": ["hm"]},
                {"label": "jc", "query_ids": [544, 83], "query_tokens": ["Ġj", "c"], "source_ids": [90, 83], "source_tokens": ["j", "c"]},
                {"label": "on", "query_ids": [335], "query_tokens": ["Ġon"], "source_ids": [258], "source_tokens": ["on"]},
            ],
        },
        "in_memory_indented_variants_of_same_256_rows": {
            "n": 256, "query_source_same_token_ids": 256,
            "prompt_tokens": {"min": 178, "mean": 192.51171875, "max": 213},
            "transform": "Exactly one ASCII space before every edge-source line; all facts, IDs, query depth, edge order and answer positions fixed",
        },
        "node_A_through_Y_context_checks": {
            "identifiers_checked": 25, "varying_letter_suffix_atomic_in_query_source_and_option": 25,
            "varying_letter_suffix_same_ids_across_contexts": 25,
            "entire_Node_letter_identifier_atomic": 0,
            "example_A": {"query_identifier_ids": [22203, 330], "source_identifier_ids": [12176, 330],
                          "varying_suffix_ids_all_contexts": [330]},
            "interpretation": "Node A is two tokens. Only the varying A suffix is a context-stable single token.",
        },
        "in_memory_Node_variants_of_same_256_rows": {
            "n": 256, "query_source_same_suffix_ids": 256,
            "prompt_tokens": {"min": 220, "mean": 220, "max": 220},
            "mapping": "Sorted original node labels mapped bijectively to Node A through Node Y; graph, hop count, edge order and answer positions preserved",
        },
        "limits": [
            "Measurements concern the 256 sampled training rows, not direct tokenizer measurements of all 512 paired development prompts",
            "Different tokenization is not evidence that it caused the training failure",
            "Indentation also changes prompt length and formatting; compute and model-format familiarity are possible confounds",
            "Node renaming additionally changes vocabulary size, token familiarity and total prompt length",
            "No accuracy or deeper-loop benefit was measured by this audit",
        ],
    }


def prepare_format_probe(root: str | Path) -> dict:
    root = Path(root).resolve()
    source_path = root / "data/diagnostic-onehop/dev.jsonl"
    source = _rows(source_path)
    if len(source) != 512:
        raise ValueError("This requested paired probe must contain the existing 512 development examples")
    for name in (ORIGINAL, INDENTED):
        if (root / "data" / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing probe {name}; use --verify-only")
    variants = {ORIGINAL: copy.deepcopy(source), INDENTED: copy.deepcopy(source)}
    for row in source:
        verify_row(row)
    for row in variants[INDENTED]:
        row["prompt"] = indent_prompt(row["prompt"])
    for name, variant in ((ORIGINAL, "original"), (INDENTED, "indented")):
        directory = root / "data" / name
        _write_rows(directory / "dev.jsonl", variants[name])
        manifest = {
            "manifest_version": 1, "dataset_type": "paired_development_format_probe", "variant": variant,
            "source": str(source_path), "count": len(source), "split": "dev", "train_split_present": False,
            "family": "pointer_chasing", "difficulty": 1, "node_count": 25,
            "pair_key": "id", "paired_dataset": INDENTED if variant == "original" else ORIGINAL,
            "same_ids_answers_facts_edge_order_option_mapping_and_row_order": True,
            "prompt_transform": "none" if variant == "original" else "add exactly one ASCII space before each of 25 edge-source lines",
            "prompt_characters": 387 if variant == "original" else 412,
            "independent_new_examples": 0,
            "evaluation_role": "Two views of the same already-held-out diagnostic dev instances; paired comparison, not new independent data",
            "verification": "Original solver verifies original rows and the indented rows after exact removal of only introduced edge-source spaces",
            "limits": ["Development-only format comparison", "No training split", "No claim that tokenization alone causes a performance difference",
                       "No inference that a format effect establishes useful deeper recurrence"],
        }
        _write_json(directory / "manifest.json", manifest)
    verification = verify_format_probe(root)
    audit = tokenizer_audit_record()
    audit["prepared_development_probe"] = {"original": str(root / "data" / ORIGINAL),
                                           "indented": str(root / "data" / INDENTED), **verification}
    _write_json(root / "artifacts/tokenization-audit.json", audit)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_format_probe(args.root) if args.verify_only else prepare_format_probe(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
