"""Prepare traceable, document-disjoint data for fresh S5 distillation.

The split is assigned to normalized complete questions/documents BEFORE token
chunks are made. Calibration is reserved from the training side, never dev.
JSONL records are unpadded; ``collate_records`` provides explicit masks when a
consumer needs padding. ``*_prompts.jsonl`` keeps complete math questions for
on-policy generation, independently of trajectory chunking.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import unicodedata

import numpy as np


SOURCES = {
    "openr1": {"repo": "open-r1/OpenR1-Math-220k", "config": "default",
               "revision": "e4e141ec9dea9f8326f4d347be56105859b2bd68", "split": "train"},
    "fineweb": {"repo": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
                "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "split": "train"},
}


def normalize_text(text: str) -> str:
    """Conservative normalized exact matching, not semantic decontamination."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def assign_split(document_hash: str, seed: int, dev_fraction: float = 0.02,
                 calibration_fraction: float = 0.01) -> str:
    """Two independent deterministic draws; calibration is train-side only."""
    if not 0 <= dev_fraction < 1 or not 0 <= calibration_fraction < 1:
        raise ValueError("Split fractions must lie in [0, 1).")

    def draw(label):
        value = hashlib.sha256(f"{seed}:{label}:{document_hash}".encode()).digest()
        return int.from_bytes(value[:8], "big") / 2 ** 64

    if draw("dev") < dev_fraction:
        return "dev"
    return "calibration" if draw("train-calibration") < calibration_fraction else "train"


def first_verified_trace(row: dict) -> tuple[str, int | None]:
    """Use the first complete verified R1 trace; never silently substitute an answer."""
    generations = row.get("generations") or []
    verified = row.get("correctness_math_verify") or []
    judged = row.get("correctness_llama") or []
    complete = row.get("is_reasoning_complete") or []
    for i, trace in enumerate(generations):
        correct = (i < len(verified) and verified[i] is True) or (i < len(judged) and judged[i] is True)
        finished = not complete or (i < len(complete) and complete[i] is True)
        if trace and correct and finished:
            return trace, i
    return "", None


def make_document(row: dict, source: str, tokenizer) -> dict | None:
    if source == "openr1":
        problem = (row.get("problem") or "").strip()
        trace, trace_index = first_verified_trace(row)
        if not problem or not trace:
            return None
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        trace_ids = tokenizer.encode(trace, add_special_tokens=False)
        # Keep the user prompt boundary exact, independent of BPE joining rules.
        token_ids = prompt_ids + trace_ids + [tokenizer.eos_token_id]
        identity = text_hash(problem)
        return {"document_id": f"openr1:{identity}", "source": source,
                "source_row_id": str(row.get("uuid") or identity),
                "content_sha256": identity, "raw_source": row.get("source"),
                "trace_index": trace_index, "input_ids": token_ids,
                "prompt_ids": prompt_ids, "question_sha256": identity,
                "question_text": problem}
    if source != "fineweb":
        raise ValueError(f"Unknown source {source}")
    body = (row.get("text") or "").strip()
    if not body:
        return None
    identity = text_hash(body)
    token_ids = tokenizer.encode(body, add_special_tokens=False) + [tokenizer.eos_token_id]
    return {"document_id": f"fineweb:{identity}", "source": source,
            "source_row_id": str(row.get("id") or identity), "content_sha256": identity,
            "input_ids": token_ids, "prompt_ids": None, "source_url": row.get("url")}


def chunk_document(document: dict, split: str, chunk_length: int = 2048,
                   prefix_length: int = 128, min_length: int = 64,
                   max_document_tokens: int = 16384) -> list[dict]:
    """Chunk only after split assignment. Never concatenate different documents."""
    if not 0 < prefix_length < chunk_length or min_length < 2:
        raise ValueError("Require 0 < prefix_length < chunk_length and min_length >= 2.")
    ids = document["input_ids"][:max_document_tokens]
    common = {k: document[k] for k in ("document_id", "source", "source_row_id", "content_sha256")}
    records = []
    for chunk_index, start in enumerate(range(0, len(ids), chunk_length)):
        chunk = ids[start:start + chunk_length]
        if len(chunk) < min_length:
            continue
        complete_prompt = document.get("prompt_ids") if start == 0 else None
        eligible = bool(complete_prompt and len(complete_prompt) < len(chunk))
        prompt_len = len(complete_prompt) if eligible else min(prefix_length, len(chunk) - 1)
        records.append(common | {
            "record_id": f"{document['document_id']}:{chunk_index}", "split": split,
            "input_ids": chunk, "valid_length": len(chunk), "prompt_len": prompt_len,
            "prompt_ids": complete_prompt if eligible else None,
            "eligible_on_policy": eligible, "chunk_index": chunk_index, "token_start": start,
        })
    return records


def collate_records(records: list[dict], pad_token_id: int = 0,
                    prompt_length: int | None = None) -> dict[str, np.ndarray]:
    """Return masks for input positions (logit at i predicts token i+1).

    continuation_mask starts at P-1 so the last prompt logit supervises the
    first continuation token; the last actual token has no next-token label.
    """
    if not records:
        raise ValueError("Empty batch")
    lengths = np.asarray([len(r["input_ids"]) for r in records], dtype=np.int64)
    prompts = np.asarray([prompt_length if prompt_length is not None else r["prompt_len"] for r in records], dtype=np.int64)
    if np.any(prompts < 1) or np.any(prompts >= lengths):
        raise ValueError("Every prompt must be nonempty and leave at least one continuation token")
    width = int(lengths.max())
    ids = np.full((len(records), width), pad_token_id, dtype=np.int64)
    for row, record in enumerate(records):
        ids[row, :lengths[row]] = record["input_ids"]
    positions = np.arange(width)[None, :]
    attention_mask = positions < lengths[:, None]
    next_token_mask = positions < lengths[:, None] - 1
    continuation_mask = next_token_mask & (positions >= prompts[:, None] - 1)
    return {"input_ids": ids, "attention_mask": attention_mask,
            "continuation_mask": continuation_mask, "next_token_mask": next_token_mask,
            "valid_lengths": lengths, "prompt_lengths": prompts}


def iter_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_records(directory: str | Path, split: str = "train", prompts: bool = False) -> list[dict]:
    if split not in {"train", "calibration", "dev"}:
        raise ValueError(split)
    suffix = "_prompts" if prompts else ""
    return list(iter_jsonl(Path(directory) / f"{split}{suffix}.jsonl"))


def prepare(args) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=Path(args.tokenizer).exists())
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs EOS token")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or any(output.glob("*.jsonl")):
        raise FileExistsError(f"Refusing to overwrite a prepared or partial dataset: {output}")
    excluded_hashes = set()
    for path in args.exclude_jsonl:
        for record in iter_jsonl(path):
            question = record.get("problem") or record.get("question") or record.get("prompt")
            if isinstance(question, str):
                excluded_hashes.add(text_hash(question))
    handles = {name: open(output / f"{name}.jsonl", "w", encoding="utf-8")
               for name in ("train", "calibration", "dev", "train_prompts", "calibration_prompts", "dev_prompts", "documents")}
    counts = Counter()
    seen_hashes = set()
    seen_source_rows = {s: set() for s in SOURCES}
    source_rows = {}
    actual_sources = {}
    try:
        for source, spec in SOURCES.items():
            limit = getattr(args, f"{source}_docs")
            raw_path = getattr(args, f"{source}_jsonl")
            if raw_path:
                rows = iter_jsonl(raw_path)
                provenance = {"local_jsonl": str(Path(raw_path).resolve()),
                              "sha256": hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
                              "upstream_revision": "not asserted for a local override"}
            else:
                rows = load_dataset(spec["repo"], spec["config"], split=spec["split"],
                                    revision=spec["revision"], streaming=True)
                provenance = spec
                # Prefix selection is reproducible and explicitly recorded. No stochastic
                # streaming shuffle whose buffer could hide the true source boundary.
            actual_sources[source] = provenance
            consumed = 0
            for row in rows:
                if counts[f"documents/{source}"] >= limit:
                    break
                consumed += 1
                document = make_document(row, source, tokenizer)
                if document is None:
                    counts[f"skipped/{source}/missing_verified_content"] += 1
                    continue
                identity = document["content_sha256"]
                if identity in excluded_hashes:
                    counts[f"skipped/{source}/benchmark_exact_match"] += 1
                    continue
                row_id = document["source_row_id"]
                if identity in seen_hashes or row_id in seen_source_rows[source]:
                    counts[f"skipped/{source}/duplicate"] += 1
                    continue
                split = assign_split(identity, args.seed, args.dev_fraction, args.calibration_fraction)
                records = chunk_document(document, split, args.chunk_length, args.prefix_length,
                                         args.min_length, args.max_document_tokens)
                if not records:
                    counts[f"skipped/{source}/too_short"] += 1
                    continue
                seen_hashes.add(identity)
                seen_source_rows[source].add(row_id)
                counts[f"documents/{source}"] += 1
                counts[f"documents/{split}/{source}"] += 1
                for record in records:
                    handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
                    counts[f"records/{split}/{source}"] += 1
                    counts[f"tokens/{split}/{source}"] += record["valid_length"]
                prompt_ids = document.get("prompt_ids")
                if source == "fineweb":
                    prompt_ids = document["input_ids"][:min(args.prefix_length, len(document["input_ids"]) - 1)]
                if prompt_ids and len(prompt_ids) <= args.max_prompt_length:
                    prompt_record = {k: document[k] for k in ("document_id", "source", "source_row_id", "content_sha256")}
                    prompt_record |= {"record_id": document["document_id"] + ":prompt", "split": split,
                                      "input_ids": prompt_ids, "prompt_ids": prompt_ids,
                                      "prompt_len": len(prompt_ids), "valid_length": len(prompt_ids),
                                      "complete_question": source == "openr1"}
                    handles[split + "_prompts"].write(json.dumps(prompt_record, separators=(",", ":")) + "\n")
                    counts[f"prompts/{split}/{source}"] += 1
                elif source == "openr1":
                    counts[f"skipped_prompts/{split}/question_too_long"] += 1
                meta = {k: v for k, v in document.items() if k not in {"input_ids", "prompt_ids", "question_text"}}
                meta |= {"split": split, "source_spec": provenance, "source_row_number": consumed - 1,
                         "original_token_length": len(document["input_ids"]), "kept_token_length": sum(r["valid_length"] for r in records),
                         "full_prompt_length": len(document.get("prompt_ids") or []), "record_count": len(records)}
                handles["documents"].write(json.dumps(meta, separators=(",", ":")) + "\n")
                if counts[f"documents/{source}"] % 1000 == 0:
                    print(json.dumps({"PREP_PROGRESS": {"source": source, "documents": counts[f"documents/{source}"], "source_rows": consumed}}), flush=True)
            source_rows[source] = consumed
            if counts[f"documents/{source}"] < limit:
                raise RuntimeError(f"{source} ended before requested {limit} usable documents")
    finally:
        for handle in handles.values():
            handle.close()
    for split in ("train", "calibration", "dev"):
        if not sum(counts[f"records/{split}/{s}"] for s in SOURCES):
            raise RuntimeError(f"Empty {split}; increase document pool")
    tokenizer_path = Path(args.tokenizer)
    tokenizer_hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        path = tokenizer_path / name
        if path.is_file():
            tokenizer_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "sources": actual_sources, "source_selection": "first N usable unique documents in source order",
                "source_rows_consumed": source_rows, "args": vars(args), "counts": dict(sorted(counts.items())),
                "tokenizer_sha256": tokenizer_hashes, "tokenizer_vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id,
                "split_unit": "normalized full math question / normalized full web document before chunking",
                "calibration": "reserved subset of train-side document hash split; disjoint from optimization train and dev",
                "deduplication": "normalized exact content/question hash and source row id; no semantic or substring claims",
                "benchmark_exclusion": {"files": args.exclude_jsonl, "normalized_exact_question_hashes": len(excluded_hashes),
                                        "limitation": "Only supplied benchmark question text; no fuzzy, paraphrase, or answer matching"},
                "source_mix": "Trainer must sample source-aware for 60% openr1 / 40% fineweb; physical token ratio is reported, not forced",
                "versions": {name: importlib.metadata.version(name) for name in ("datasets", "transformers", "tokenizers", "numpy")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"PREP_DONE": manifest}), flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--openr1-docs", type=int, default=10000)
    parser.add_argument("--fineweb-docs", type=int, default=24000)
    parser.add_argument("--openr1-jsonl", default=None, help="Optional raw local rows instead of pinned HF streaming")
    parser.add_argument("--fineweb-jsonl", default=None, help="Optional raw local rows instead of pinned HF streaming")
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--dev-fraction", type=float, default=0.02)
    parser.add_argument("--calibration-fraction", type=float, default=0.01)
    parser.add_argument("--chunk-length", type=int, default=2048)
    parser.add_argument("--prefix-length", type=int, default=128)
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--max-document-tokens", type=int, default=16384)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
