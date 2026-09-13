"""Semantic, split-integrity and confound checks for generated reasoning data."""

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import generate_dataset, solve_prompt, verify_dataset, verify_row


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ouro-data-test-")
        cls.directory = Path(cls.temporary.name) / "data"
        cls.counts = {"train": 384, "dev": 96, "test": 96, "ood": 96}
        cls.manifest = generate_dataset(
            cls.directory, train_count=384, dev_count=96, test_count=96, ood_count=96, seed=815,
        )
        cls.rows = {
            split: [json.loads(line) for line in (cls.directory / f"{split}.jsonl").read_text().splitlines()]
            for split in cls.counts
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_verifier_and_split_integrity(self):
        result = verify_dataset(self.directory)
        self.assertEqual(result["verified_rows"], sum(self.counts.values()))
        self.assertEqual(result["instance_overlap"], 0)
        keys = set()
        for split, rows in self.rows.items():
            self.assertEqual(len(rows), self.counts[split])
            for row in rows:
                self.assertEqual(row["split"], split)
                self.assertNotIn(row["metadata"]["instance_key"], keys)
                keys.add(row["metadata"]["instance_key"])
                self.assertIn(row["difficulty"], (10, 12) if split == "ood" else (1, 2, 3, 4, 6, 8))

    def test_repeat_generation_is_byte_identical(self):
        duplicate = Path(self.temporary.name) / "repeat"
        generate_dataset(duplicate, train_count=384, dev_count=96, test_count=96, ood_count=96, seed=815)
        for filename in ("train.jsonl", "dev.jsonl", "test.jsonl", "ood.jsonl", "manifest.json"):
            self.assertEqual((self.directory / filename).read_bytes(), (duplicate / filename).read_bytes())

    def test_heldout_count_changes_do_not_change_training(self):
        other = Path(self.temporary.name) / "different_test_count"
        generate_dataset(other, train_count=384, dev_count=96, test_count=23, ood_count=19, seed=815)
        self.assertEqual((self.directory / "train.jsonl").read_bytes(), (other / "train.jsonl").read_bytes())

    def test_different_seeds_produce_disjoint_instances(self):
        other = Path(self.temporary.name) / "different_seed"
        generate_dataset(other, train_count=12, dev_count=0, test_count=0, ood_count=0, seed=816)
        existing = {row["metadata"]["instance_key"] for rows in self.rows.values() for row in rows}
        alternate = {json.loads(line)["metadata"]["instance_key"] for line in (other / "train.jsonl").read_text().splitlines()}
        self.assertTrue(existing.isdisjoint(alternate))

    def test_answer_balance_within_family_and_depth(self):
        for rows in self.rows.values():
            groups = defaultdict(Counter)
            for row in rows:
                groups[(row["family"], row["difficulty"])][row["answer"]] += 1
            for counts in groups.values():
                values = [counts[letter] for letter in "ABCDEFGH"]
                self.assertLessEqual(max(values) - min(values), 1)

    def test_nondivisible_counts_preserve_allocation_and_letter_balance(self):
        other = Path(self.temporary.name) / "nondivisible"
        manifest = generate_dataset(other, train_count=131, dev_count=7, test_count=5, ood_count=11, seed=19)
        self.assertEqual(verify_dataset(other)["verified_rows"], 154)
        for split in ("train", "dev", "test", "ood"):
            depths = (10, 12) if split == "ood" else (1, 2, 3, 4, 6, 8)
            strata = manifest["splits"][split]["strata"]
            counts = [strata.get(f"{family}/d{depth}", {"count": 0})["count"] for family in ("pointer_chasing", "modular_arithmetic") for depth in depths]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_context_length_is_controlled_across_difficulties(self):
        lengths = defaultdict(set)
        for rows in self.rows.values():
            for row in rows:
                self.assertTrue(row["prompt"].endswith("\nAnswer:"))
                if row["family"] == "pointer_chasing":
                    self.assertEqual(row["metadata"]["context_size"], 25)
                    self.assertEqual(len(row["metadata"]["facts"]["edges"]), 25)
                    self.assertGreater(25 - row["difficulty"], row["difficulty"])
                    # The only character-count difference is the number of digits in k.
                    normalized = len(row["prompt"]) - len(str(row["difficulty"]))
                else:
                    self.assertEqual(row["metadata"]["context_size"], 16)
                    self.assertEqual(len(row["metadata"]["facts"]["equations"]), 16)
                    normalized = len(row["prompt"])
                lengths[row["family"]].add(normalized)
        self.assertEqual({family: len(values) for family, values in lengths.items()}, {
            "pointer_chasing": 1, "modular_arithmetic": 1,
        })

    def test_verifier_rejects_wrong_labels_and_metadata(self):
        original = self.rows["train"][0]
        for mutation in ("answer", "answer_value", "difficulty", "instance_key", "trace"):
            with self.subTest(mutation=mutation):
                row = copy.deepcopy(original)
                if mutation == "answer":
                    row["answer"] = "B" if row["answer"] == "A" else "A"
                elif mutation == "answer_value":
                    row["answer_value"] = "wrong"
                elif mutation == "difficulty":
                    row["difficulty"] += 1
                elif mutation == "instance_key":
                    row["metadata"]["instance_key"] = "fabricated"
                else:
                    row["metadata"]["construction_trace"] = []
                with self.assertRaises(ValueError):
                    verify_row(row)

    def test_semantic_identity_ignores_statement_order(self):
        for family in ("pointer_chasing", "modular_arithmetic"):
            row = copy.deepcopy(next(row for row in self.rows["train"] if row["family"] == family))
            lines = row["prompt"].splitlines()
            start = 2
            end = start + (25 if family == "pointer_chasing" else 17)
            lines[start:end] = reversed(lines[start:end])
            row["prompt"] = "\n".join(lines)
            # Neither the stored semantic id nor metadata needed to change.
            verify_row(row)

    def test_verifier_reads_prompt_rather_than_trusting_metadata(self):
        row = copy.deepcopy(next(row for row in self.rows["train"] if row["family"] == "modular_arithmetic"))
        base, value = row["metadata"]["facts"]["base"]
        row["prompt"] = row["prompt"].replace(f"{base} = {value:02d}", f"{base} = {(value + 1) % 17:02d}")
        with self.assertRaises(ValueError):
            verify_row(row)

    def test_cross_split_duplicate_is_rejected_even_with_changed_split_field(self):
        duplicate_dir = Path(self.temporary.name) / "duplicate"
        duplicate_dir.mkdir(exist_ok=True)
        for filename in ("train.jsonl", "dev.jsonl", "test.jsonl", "ood.jsonl", "manifest.json"):
            (duplicate_dir / filename).write_bytes((self.directory / filename).read_bytes())
        duplicated = copy.deepcopy(self.rows["train"][0])
        duplicated["split"] = "dev"
        (duplicate_dir / "dev.jsonl").write_text(json.dumps(duplicated) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate underlying instance"):
            verify_dataset(duplicate_dir)

    def test_independent_golden_pointer_answer(self):
        prompt = (
            "Follow exactly 5 directed links from aa. Every node has one outgoing link.\nLinks:\n"
            "dd -> ee\naa -> bb\ngg -> hh\ncc -> dd\nhh -> aa\nff -> gg\nbb -> cc\nee -> ff\n"
            "Which node do you reach?\n"
            "A) aa\nB) bb\nC) cc\nD) dd\nE) ee\nF) ff\nG) gg\nH) hh\nAnswer:"
        )
        solved = solve_prompt(prompt, "pointer_chasing")
        self.assertEqual((solved["answer"], solved["answer_value"], solved["difficulty"]), ("F", "ff", 5))

    def test_independent_golden_arithmetic_answer_and_depth(self):
        # aa=4; bb=3*4+1=13; cc=2*13+6=15 (mod 17); dd=5*15+4=11 (mod 17).
        prompt = (
            "All values are integers modulo 17. Each equation defines its left-hand variable.\nEquations:\n"
            "cc = (02 * bb + 06) mod 17\naa = 04\ndd = (05 * cc + 04) mod 17\nbb = (03 * aa + 01) mod 17\n"
            "What is the value of dd?\n"
            "A) 00\nB) 02\nC) 04\nD) 06\nE) 08\nF) 10\nG) 11\nH) 13\nAnswer:"
        )
        solved = solve_prompt(prompt, "modular_arithmetic")
        self.assertEqual((solved["answer"], solved["answer_value"], solved["difficulty"]), ("G", 11, 3))
        with self.assertRaisesRegex(ValueError, "Cycle"):
            solve_prompt(prompt.replace("03 * aa", "03 * dd"), "modular_arithmetic")

    def test_invalid_difficulty_configuration_is_rejected(self):
        invalid = [
            {"ood_difficulties": (8, 10)}, {"context_size": 12},
            {"train_difficulties": (1, 1)}, {"train_count": -1},
            {"train_difficulties": (1, 8), "ood_difficulties": (4, 6)},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_dataset(Path(self.temporary.name) / "invalid", **kwargs)


if __name__ == "__main__":
    unittest.main()
