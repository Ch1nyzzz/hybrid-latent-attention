import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ouro_depth.latent.prepare_recipe_data import (
    assign_split, chunk_document, collate_records, first_verified_trace, make_document, prepare, text_hash,
)


class ToyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 1 for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "USER:" + messages[0]["content"] + "\nASSISTANT:"

    def __len__(self):
        return 256


def math_row(problem="What is 2 + 3?"):
    return {"problem": problem, "uuid": "example", "generations": ["Wrong", "Reasoning. " * 30],
            "correctness_math_verify": [False, True], "is_reasoning_complete": [True, True]}


def test_question_split_before_chunks_and_normalized_duplicate():
    identity = text_hash("  The  same\nquestion ")
    assert identity == text_hash("the same question")
    doc = make_document(math_row(), "openr1", ToyTokenizer())
    split = assign_split(doc["content_sha256"], 17)
    chunks = chunk_document(doc, split, chunk_length=64, prefix_length=8, min_length=8)
    assert len(chunks) > 2
    assert {r["split"] for r in chunks} == {split}
    assert {r["document_id"] for r in chunks} == {doc["document_id"]}
    assert [r["token_start"] for r in chunks] == list(range(0, 64 * len(chunks), 64))


def test_split_is_deterministic_disjoint_and_calibration_from_train_side():
    identities = [text_hash(f"doc {i}") for i in range(10000)]
    observed = {s: set() for s in ("train", "calibration", "dev")}
    for identity in identities:
        split = assign_split(identity, 12, 0.1, 0.1)
        assert split == assign_split(identity, 12, 0.1, 0.1)
        observed[split].add(identity)
        no_calibration = assign_split(identity, 12, 0.1, 0)
        assert no_calibration == ("dev" if split == "dev" else "train")
    assert all(observed.values())
    assert not observed["train"] & observed["dev"]
    assert not observed["calibration"] & observed["dev"]
    assert not observed["calibration"] & observed["train"]


def test_complete_math_question_survives_and_only_initial_chunk_is_eligible():
    tokenizer = ToyTokenizer()
    document = make_document(math_row(), "openr1", tokenizer)
    expected = tokenizer.encode("USER:What is 2 + 3?\nASSISTANT:")
    assert document["prompt_ids"] == expected
    chunks = chunk_document(document, "train", chunk_length=64, prefix_length=8, min_length=8)
    assert chunks[0]["prompt_ids"] == expected
    assert chunks[0]["input_ids"][:len(expected)] == expected
    assert chunks[0]["prompt_len"] == len(expected)
    assert chunks[0]["eligible_on_policy"]
    assert all(not r["eligible_on_policy"] and r["prompt_ids"] is None for r in chunks[1:])


def test_long_math_question_is_not_truncated_into_on_policy_prompt():
    document = make_document(math_row("A very long question. " * 50), "openr1", ToyTokenizer())
    chunks = chunk_document(document, "train", chunk_length=64, prefix_length=8, min_length=8)
    assert len(document["prompt_ids"]) > 64
    assert all(not r["eligible_on_policy"] for r in chunks)
    assert document["input_ids"][:len(document["prompt_ids"])] == document["prompt_ids"]


def test_padding_and_prediction_masks_do_not_train_padding_or_prompt():
    records = [{"input_ids": [1, 2, 3, 4, 5], "prompt_len": 3},
               {"input_ids": [6, 7, 8], "prompt_len": 2}]
    batch = collate_records(records, pad_token_id=0)
    np.testing.assert_array_equal(batch["input_ids"], [[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    np.testing.assert_array_equal(batch["attention_mask"], [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    np.testing.assert_array_equal(batch["continuation_mask"], [[0, 0, 1, 1, 0], [0, 1, 0, 0, 0]])
    np.testing.assert_array_equal(batch["next_token_mask"], [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    with pytest.raises(ValueError):
        collate_records(records, prompt_length=3)


def test_only_verified_complete_trace_is_used():
    assert first_verified_trace(math_row())[1] == 1
    row = math_row()
    row["is_reasoning_complete"] = [True, False]
    assert first_verified_trace(row) == ("", None)
    assert make_document(row, "openr1", ToyTokenizer()) is None


def test_preparation_excludes_benchmark_and_records_local_override_provenance(tmp_path, monkeypatch):
    math_rows = [math_row(f"Question {i}: compute {i} + 3.") | {"uuid": str(i)} for i in range(100)]
    math_rows.insert(2, dict(math_rows[1]))
    web_rows = [{"id": str(i), "text": f"Document {i}. " + "Education and arithmetic. " * 10} for i in range(100)]
    math_file, web_file, excluded = [tmp_path / name for name in ("math.jsonl", "web.jsonl", "excluded.jsonl")]
    for path, records in ((math_file, math_rows), (web_file, web_rows), (excluded, [math_rows[0]])):
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: pytest.fail("network not permitted")))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: ToyTokenizer())))
    args = SimpleNamespace(tokenizer=str(tmp_path / "tokenizer"), output=str(tmp_path / "prepared"),
                           openr1_docs=80, fineweb_docs=80, openr1_jsonl=str(math_file), fineweb_jsonl=str(web_file),
                           exclude_jsonl=[str(excluded)], seed=123, dev_fraction=0.2, calibration_fraction=0.2,
                           chunk_length=64, prefix_length=8, min_length=8, max_document_tokens=512, max_prompt_length=128)
    manifest = prepare(args)
    assert manifest["counts"]["skipped/openr1/benchmark_exact_match"] == 1
    assert manifest["counts"]["skipped/openr1/duplicate"] == 1
    assert manifest["sources"]["openr1"]["upstream_revision"].startswith("not asserted")
    assert len(manifest["sources"]["openr1"]["sha256"]) == 64
    documents = [json.loads(line) for line in (tmp_path / "prepared/documents.jsonl").read_text().splitlines()]
    assert len(documents) == 160
    assert len({doc["document_id"] for doc in documents}) == 160
    assert text_hash(math_rows[0]["problem"]) not in {doc["content_sha256"] for doc in documents}
