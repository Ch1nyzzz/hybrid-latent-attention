"""Actual tiny-Ouro/AdamW interrupted-versus-continuous trainer resume check."""

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ouro_depth import train as trainer
from ouro_depth.model import OuroDepthModel
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


class _TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" "):
            return [2 + "ABCDEFGH".index(text[1:])]
        prompt, separator, answer = text.partition(" ")
        number = int(prompt.removeprefix("example").removesuffix(":"))
        tokens = [1, 11 + number, 20 + number]
        return tokens + ([2 + "ABCDEFGH".index(answer)] if separator else [])


class _InterruptedAfterCheckpoint(Exception):
    pass


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logit_table", logits.to(torch.bfloat16))

    def forward(self, input_ids, attention_mask, depths):
        # Row identifiers in the artificial prompts work across eval minibatches.
        logits = self.logit_table[input_ids[:, 0] - 1]
        return {depth: logits for depth in depths}


class EvaluationTieTests(unittest.TestCase):
    # Alphabetic option order intentionally differs from vocabulary/token order.
    answer_ids = [10, 2, 9, 4, 8, 6, 7, 5]

    def run_evaluation(self, logits, answer_letters):
        model = _FixedLogitModel(logits).train()
        encoded = [
            {"ids": [i + 1], "target": self.answer_ids["ABCDEFGH".index(letter)],
             "row": {"id": f"tie-{i}", "family": "pointer_chasing", "difficulty": 6, "answer": letter}}
            for i, letter in enumerate(answer_letters)
        ]
        args = SimpleNamespace(device="cpu", eval_batch=3, pad_id=0)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "evaluation"
            result = trainer.evaluate(model, encoded, self.answer_ids, args, [4, 8], prefix)
            rows = [json.loads(line) for line in Path(str(prefix) + ".predictions.jsonl").read_text().splitlines()]
        self.assertTrue(model.training)
        self.assertEqual(result["evaluator_version"], 2)
        self.assertEqual(result["choice_tie_break"], "ascending_token_id")
        for row in rows:
            for score in row["scores"].values():
                self.assertFalse(score["correct"] and not score["choice_correct"])
        return result, rows

    def test_bfloat16_ties_use_token_order_and_fractional_credit(self):
        logits = torch.zeros(6, 12)
        # Distinct FP32 scores round to the same BF16 value. Old alphabetic
        # tie-breaking picked A, while full-vocabulary argmax correctly picked B.
        logits[0, 10], logits[0, 2] = 5.01, 5.012
        logits[1, [10, 2]] = 5
        logits[2, [10, 9, 4]] = 5
        logits[3, self.answer_ids] = 5
        logits[4, [10, 2]] = 5  # The target E is outside the set of maxima.
        logits[5, [0, 10, 2]] = 5  # A non-answer wins unrestricted argmax.
        result, rows = self.run_evaluation(logits, ["B", "A", "C", "H", "E", "B"])
        expected_choices = ["B", "B", "D", "B", "B", "B"]
        expected_fractional = [0.5, 0.5, 1 / 3, 1 / 8, 0.0, 0.5]
        for depth in ("4", "8"):
            scores = [row["scores"][depth] for row in rows]
            self.assertEqual([score["choice"] for score in scores], expected_choices)
            self.assertEqual(sum(score["choice_tied"] for score in scores), 6)
            self.assertEqual([score["correct"] for score in scores], [True, False, False, False, False, False])
            self.assertEqual([score["choice_correct"] for score in scores], [True, False, False, False, False, True])
            for score, expected in zip(scores, expected_fractional):
                self.assertAlmostEqual(score["choice_tie_aware_correct"], expected, places=7)
            summary = result["metrics"]["all"]["by_depth"][depth]
            self.assertEqual(summary["choice_tie_rate"], 1.0)
            self.assertAlmostEqual(summary["accuracy"], 1 / 6)
            self.assertAlmostEqual(summary["choice_accuracy"], 2 / 6)
            self.assertAlmostEqual(summary["choice_tie_aware_accuracy"], sum(expected_fractional) / 6, places=7)

    def test_unique_choice_maxima_preserve_predictions_and_full_credit(self):
        logits = torch.zeros(3, 12)
        logits[0, 9] = 5  # Correct C, unique global maximum.
        logits[1, 4] = 5  # Wrong D, unique global maximum; correct label is A.
        logits[2, 7] = 5  # Correct G within choices, but an outside token wins.
        logits[2, 0] = 6
        result, rows = self.run_evaluation(logits, ["C", "A", "G"])
        for depth in ("4", "8"):
            scores = [row["scores"][depth] for row in rows]
            self.assertEqual([score["choice"] for score in scores], ["C", "D", "G"])
            self.assertEqual([score["prediction_token"] for score in scores], [9, 4, 0])
            self.assertEqual([score["choice_tied"] for score in scores], [False, False, False])
            self.assertEqual([score["choice_tie_aware_correct"] for score in scores], [1.0, 0.0, 1.0])
            summary = result["metrics"]["all"]["by_depth"][depth]
            self.assertEqual(summary["choice_tie_rate"], 0.0)
            self.assertAlmostEqual(summary["accuracy"], 1 / 3)
            self.assertAlmostEqual(summary["choice_accuracy"], 2 / 3)
            self.assertAlmostEqual(summary["choice_tie_aware_accuracy"], 2 / 3)


class TrainerResumeTests(unittest.TestCase):
    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_cpu_resume_restores_optimizer_rng_order_and_rejects_wrong_identity(self):
        """Dropout makes a missing RNG restore observable in the resumed weights."""
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            stack.callback(torch.set_num_threads, old_threads)
            # Keep the test CPU-only even when run on a host with visible GPUs.
            stack.enter_context(patch.object(torch.cuda, "get_rng_state_all", return_value=[]))
            stack.enter_context(patch.object(torch.cuda, "set_rng_state_all"))
            stack.enter_context(patch.object(torch.cuda, "manual_seed_all"))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            stack.enter_context(patch.object(trainer, "evaluate", return_value={"metrics": {}, "count": 3}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            data = root / "data"
            data.mkdir()
            rows = [
                {"id": f"row-{i}", "prompt": f"example{i}:", "answer": "ABCDEFGH"[i],
                 "family": "pointer_chasing", "difficulty": [1, 4, 8][i]}
                for i in range(3)
            ]
            for split in ("train", "dev"):
                (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

            torch.manual_seed(802)
            config = OuroConfig(
                vocab_size=43, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=32, attention_dropout=0.1,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
                use_cache=False, tie_word_embeddings=False,
            )
            config._attn_implementation = "sdpa"
            initial = OuroForCausalLM(config).float()

            def fresh_model():
                return OuroDepthModel(copy.deepcopy(initial), mode="full", checkpointing=True)

            def arguments(name):
                return SimpleNamespace(
                    output=str(root / name), data_dir=str(data), model_path=str(root / "same-base"),
                    arm="curriculum", seed=73, mode="full", lora_rank=4,
                    device="cpu", batch_size=2, micro_batch=1, eval_batch=2,
                    lr=1e-3, weight_decay=0.01, clip=1.0, warmup_fraction=0.05,
                    budget=1400, max_updates=4, backprop_loops=None,
                    no_checkpointing=False, train_limit=0, max_length=32,
                    dev_limit=0, pad_id=0, eval_every=0, save_every=2,
                    depths=[4, 8], resume=None,
                )

            uninterrupted_args = arguments("uninterrupted")
            uninterrupted = fresh_model()
            trainer.seed_all(uninterrupted_args.seed)
            trainer.train(uninterrupted, _TinyTokenizer(), uninterrupted_args)

            interrupted_args = arguments("interrupted")
            interrupted = fresh_model()
            original_checkpoint = trainer.checkpoint

            def save_then_interrupt(model, optimizer, output, state):
                result = original_checkpoint(model, optimizer, output, state)
                if state["update"] == 2:
                    raise _InterruptedAfterCheckpoint()
                return result

            trainer.seed_all(interrupted_args.seed)
            with patch.object(trainer, "checkpoint", side_effect=save_then_interrupt):
                with self.assertRaises(_InterruptedAfterCheckpoint):
                    trainer.train(interrupted, _TinyTokenizer(), interrupted_args)

            run_output = Path(interrupted_args.output)
            source = run_output / "checkpoint-2"
            saved_at_two = torch.load(source / "training.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved_at_two["state"]["update"], 2)
            self.assertFalse((run_output / "completed.json").exists())
            for optimizer_state in saved_at_two["optimizer"]["state"].values():
                self.assertEqual(optimizer_state["step"].item(), 2)
                self.assertTrue(torch.isfinite(optimizer_state["exp_avg"]).all())
                self.assertTrue(torch.isfinite(optimizer_state["exp_avg_sq"]).all())

            identity_before = (run_output / "identity.json").read_text()
            wrong_arm = copy.copy(interrupted_args)
            wrong_arm.resume = str(source)
            wrong_arm.arm = "fixed8"
            with self.assertRaisesRegex(ValueError, "configuration/data mismatch"):
                trainer.train(fresh_model(), _TinyTokenizer(), wrong_arm)
            self.assertEqual((run_output / "identity.json").read_text(), identity_before)

            foreign = root / "foreign" / "checkpoint-2"
            shutil.copytree(source, foreign)
            wrong_source = copy.copy(interrupted_args)
            wrong_source.resume = str(foreign)
            with self.assertRaisesRegex(ValueError, "must belong to this run output directory"):
                trainer.train(fresh_model(), _TinyTokenizer(), wrong_source)

            resumed = fresh_model()
            interrupted_args.resume = str(source)
            torch.manual_seed(9999)
            torch.randn(97)  # Restore must override a different process RNG state.
            trainer.train(resumed, _TinyTokenizer(), interrupted_args)

            continuous_state = uninterrupted.state_dict()
            self.assert_tree_equal(continuous_state, resumed.state_dict())
            uninterrupted_final = json.loads((Path(uninterrupted_args.output) / "completed.json").read_text())
            resumed_final = json.loads((run_output / "completed.json").read_text())
            self.assert_tree_equal(uninterrupted_final["state"], resumed_final["state"])
            final_step = resumed_final["state"]["update"]
            self.assertEqual(final_step, 4)
            self.assertGreater(resumed_final["state"]["epoch"], 0)  # Covered a data reshuffle.

            continuous_checkpoint = torch.load(
                Path(uninterrupted_final["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False
            )
            resumed_checkpoint = torch.load(
                Path(resumed_final["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False
            )
            self.assert_tree_equal(continuous_checkpoint, resumed_checkpoint)
            self.assertTrue(any(
                not torch.equal(saved_at_two["optimizer"]["state"][key]["exp_avg"], value["exp_avg"])
                for key, value in resumed_checkpoint["optimizer"]["state"].items()
            ))
            for optimizer_state in resumed_checkpoint["optimizer"]["state"].values():
                self.assertEqual(optimizer_state["step"].item(), final_step)
            records = [json.loads(line) for line in (run_output / "metrics.jsonl").read_text().splitlines()]
            continued_updates = [row for row in records if row["event"] == "update" and row["update"] > 2]
            self.assertEqual([row["update"] for row in continued_updates], [3, 4])
            self.assertTrue(all(math.isfinite(row["loss"]) and row["grad_norm"] > 0 for row in continued_updates))


if __name__ == "__main__":
    unittest.main()
