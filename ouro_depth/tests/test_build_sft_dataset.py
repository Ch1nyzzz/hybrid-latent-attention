"""Unit tests for the mathematical SFT data pipeline (build_sft_dataset)."""
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from ouro_depth.latent.build_sft_dataset import (
    find_best_verified_trace,
    load_benchmark_hashes,
    process_parquet_shard,
    build_sft_dataset,
)
from ouro_depth.latent.prepare_recipe_data import text_hash
from ouro_depth.latent.sft_replay import SFTDataset


TOKENIZER_PATH = Path(__file__).resolve().parents[2] / "artifacts/s5b-triton-20260915/training"


class TestBuildSFTDataset(unittest.TestCase):

    def test_find_best_verified_trace(self):
        # Case 1: unverified trace
        row_unverified = {
            "generations": ["<think>Thinking...</think>\\boxed{10}"],
            "correctness_math_verify": [False],
            "is_reasoning_complete": [True],
        }
        trace, idx = find_best_verified_trace(row_unverified)
        self.assertIsNone(trace)
        self.assertIsNone(idx)

        # Case 2: verified but incomplete
        row_incomplete = {
            "generations": ["<think>Thinking...</think>\\boxed{10}"],
            "correctness_math_verify": [True],
            "is_reasoning_complete": [False],
        }
        trace, idx = find_best_verified_trace(row_incomplete)
        self.assertIsNone(trace)

        # Case 3: verified, complete, but lacks boxed answer
        row_no_boxed = {
            "generations": ["<think>Thinking...</think> The answer is 10."],
            "correctness_math_verify": [True],
            "is_reasoning_complete": [True],
        }
        trace, idx = find_best_verified_trace(row_no_boxed)
        self.assertIsNone(trace)

        # Case 4: multiple verified traces, should pick the shortest
        long_trace = "<think>" + "long reasoning " * 50 + "</think>\\boxed{42}"
        short_trace = "<think>quick</think>\\boxed{42}"
        row_multiple = {
            "generations": [long_trace, short_trace],
            "correctness_math_verify": [True, True],
            "is_reasoning_complete": [True, True],
        }
        trace, idx = find_best_verified_trace(row_multiple)
        self.assertEqual(trace, short_trace)
        self.assertEqual(idx, 1)

    def test_benchmark_hash_collection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            jsonl_file = tmppath / "bench.jsonl"
            jsonl_file.write_text(json.dumps({"problem": "Compute 2 + 2."}) + "\n")

            hashes, loaded = load_benchmark_hashes(cache_root=tmppath, explicit_files=[str(jsonl_file)])
            expected_hash = text_hash("Compute 2 + 2.")
            self.assertIn(expected_hash, hashes)
            self.assertEqual(loaded, [str(jsonl_file)])

    def test_process_parquet_shard_and_decontamination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            parquet_path = tmppath / "test_shard.parquet"

            # Create mock parquet data
            p1 = "What is 3 + 5?"
            g1 = "<think>3 + 5 = 8</think>\\boxed{8}"
            p2 = "What is 7 * 6?"  # will be contaminated
            g2 = "<think>7 * 6 = 42</think>\\boxed{42}"
            p3 = "What is 10 - 4?"  # valid
            g3 = "<think>10 - 4 = 6</think>\\boxed{6}"

            data = {
                "problem": [p1, p2, p3],
                "uuid": ["u1", "u2", "u3"],
                "source": ["test_source", "test_source", "test_source"],
                "generations": [[g1], [g2], [g3]],
                "correctness_math_verify": [[True], [True], [True]],
                "correctness_llama": [[False], [False], [False]],
                "is_reasoning_complete": [[True], [True], [True]],
            }
            table = pa.Table.from_pydict(data)
            pq.write_table(table, parquet_path)

            # Benchmark excludes p2
            bench_hashes = {text_hash(p2)}

            candidates, counts = process_parquet_shard(
                str(parquet_path),
                benchmark_hashes=bench_hashes,
                tokenizer_path_str=str(TOKENIZER_PATH),
                max_prompt_length=1024,
                max_response_length=2048,
            )

            self.assertEqual(counts["total_source_rows"], 3)
            self.assertEqual(counts["excluded_benchmark_leak"], 1)
            self.assertEqual(counts["candidate_kept"], 2)
            self.assertEqual(len(candidates), 2)

            # Check candidate formatting
            c0 = candidates[0]
            self.assertEqual(c0["problem"], p1)
            self.assertEqual(c0["prompt_len"] + c0["response_len"], len(c0["input_ids"]))
            self.assertEqual(c0["input_ids"], c0["prompt_ids"] + c0["response_ids"])

            # Verify ending token is <|im_end|>\n
            tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
            self.assertTrue(tok.decode(c0["response_ids"]).endswith("<|im_end|>\n"))

    def test_build_sft_dataset_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            raw_dir = tmppath / "raw"
            pool_dir = raw_dir / "data"
            pool_dir.mkdir(parents=True)
            out_dir = tmppath / "sft_out"

            p1 = "Problem 1: Solve x = 1."
            g1 = "<think>x is 1</think>\\boxed{1}"
            p2 = "Problem 2: Solve y = 2."
            g2 = "<think>y is 2</think>\\boxed{2}"

            table = pa.Table.from_pydict({
                "problem": [p1, p2],
                "uuid": ["id1", "id2"],
                "source": ["src", "src"],
                "generations": [[g1], [g2]],
                "correctness_math_verify": [[True], [True]],
                "correctness_llama": [[False], [False]],
                "is_reasoning_complete": [[True], [True]],
            })
            pq.write_table(table, pool_dir / "shard0.parquet")

            manifest = build_sft_dataset(
                raw_dir=raw_dir,
                output_dir=out_dir,
                tokenizer_path=TOKENIZER_PATH,
                benchmark_cache=tmppath / "no_cache",
                pools=("data",),
                max_prompt_length=1024,
                max_response_length=2048,
                num_workers=1,
            )

            self.assertTrue((out_dir / "train.jsonl").exists())
            self.assertTrue((out_dir / "dev.jsonl").exists())
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertEqual(manifest["counts"]["candidate_kept"], 2)

            # Check that SFTDataset can load and index the output
            # We check whichever file has data (train or dev)
            target = out_dir / "train.jsonl" if out_dir / "train.jsonl" and (out_dir / "train.jsonl").stat().st_size > 0 else out_dir / "dev.jsonl"
            dataset = SFTDataset(target, max_prompt=1024, max_response=2048)
            self.assertGreater(len(dataset.offsets), 0)
            sample = dataset.sample_at(0, seed=123)
            self.assertIn("input_ids", sample)
            self.assertIn("prompt_len", sample)
            self.assertGreater(sample["prompt_len"], 0)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
