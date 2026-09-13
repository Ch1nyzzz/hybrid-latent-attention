"""Actual CPU dropout/checkpoint resume equivalence for the frozen v3 plan."""

from collections import Counter
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ouro_depth import train_v3 as trainer
from ouro_depth.model import OuroDepthModel
from ouro_depth.v3_plan import lr_multiplier
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


class _TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" "):
            return [2 + "ABCDEFGH".index(text[1:])]
        prompt, separator, answer = text.partition(" ")
        index = int(prompt.removeprefix("example").removesuffix(":"))
        # Differing prompt lengths make valid/padded-token accounting observable.
        ids = [1, 11 + index, 20 + index] + [31] * (index % 3)
        return ids + ([2 + "ABCDEFGH".index(answer)] if separator else [])


class _InterruptedAfterTwo(Exception):
    pass


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def process_rng():
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
    }


class FixedPaddingTests(unittest.TestCase):
    def test_fixed_width_masks_select_last_valid_token_and_keep_answer_targets_separate(self):
        items = [{"ids": [1, 7, 8], "target": 3}, {"ids": [1, 9], "target": 4}]
        ids, mask, targets = trainer.collate_fixed(items, pad_id=0, device="cpu", padding_width=8)
        self.assertEqual(tuple(ids.shape), (2, 8))
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(ids.device.type, "cpu")
        self.assertEqual(ids.tolist(), [[1, 7, 8, 0, 0, 0, 0, 0], [1, 9, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(targets.tolist(), [3, 4])
        last_valid = mask.sum(dim=1) - 1
        self.assertEqual(ids[torch.arange(2), last_valid].tolist(), [8, 9])
        single_ids, single_mask, _ = trainer.collate_fixed(items[1:], pad_id=0, device="cpu", padding_width=8)
        self.assertEqual(tuple(single_ids.shape), (1, 8))
        torch.testing.assert_close(single_ids[0], ids[1], rtol=0, atol=0)
        torch.testing.assert_close(single_mask[0], mask[1], rtol=0, atol=0)

    def test_overlong_input_is_rejected_instead_of_truncated(self):
        with self.assertRaises(ValueError):
            trainer.collate_fixed([{"ids": list(range(9)), "target": 3}], pad_id=0, device="cpu", padding_width=8)


class V3TrainerResumeTests(unittest.TestCase):
    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
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

    def test_zero_loss_zero_gradient_batches_complete_with_finite_optimizer_state(self):
        class SolvedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))
                self.checkpointing = True
                self.config = SimpleNamespace(num_hidden_layers=1)

            @property
            def trainable_count(self):
                return self.anchor.numel()

            def forward(self, input_ids, attention_mask, depths, backprop_loops=None):
                # Every row has answer A (token 2). The scalar keeps the logits
                # attached to autograd while giving an exactly zero gradient.
                logits = torch.full((input_ids.shape[0], 43), -1000.0, device=input_ids.device)
                logits[:, 2] = 1000.0
                logits = logits + self.anchor * 0
                return {depth: logits for depth in depths}

            def save_trainable(self, path):
                directory = Path(path)
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / "trainable.pt"
                torch.save({"state_dict": self.state_dict()}, target)
                return target

            def load_trainable(self, path):
                source = Path(path)
                if source.is_dir():
                    source = source / "trainable.pt"
                self.load_state_dict(torch.load(source, map_location="cpu", weights_only=True)["state_dict"])

        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(trainer, "evaluate", return_value={"metrics": {}, "count": 6}))
            stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU test initialized CUDA")))
            stack.enter_context(patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CPU checkpoint read CUDA RNG")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            rows = [
                {"id": f"solved-{index}", "prompt": f"example{index}:", "answer": "A",
                 "family": "pointer_chasing", "difficulty": difficulty}
                for index, difficulty in enumerate((1, 2, 3, 4, 6, 8))
            ]
            data = root / "data"
            data.mkdir()
            for split in ("train", "dev"):
                (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            model = SolvedModel()
            initializer = model.save_trainable(root / "init")
            args = SimpleNamespace(
                output=str(root / "solved"), data_dir=str(data), model_path=str(root / "same-base"),
                checkpoint=str(initializer), arm="conditional", seed=73,
                device="cpu", mode="full", lora_rank=32,
                batch_size=2, micro_batch=1, eval_batch=2,
                lr=1e-3, weight_decay=0.01, clip=1.0, warmup_fraction=0.05,
                budget=4096, max_updates=100, max_length=16, padding_width=0,
                train_limit=0, dev_limit=0, pad_id=0, eval_every=0,
                save_every=2, depths=[4, 8], resume=None, plan_path=None,
            )
            trainer.train(model, _TinyTokenizer(), args)
            output = Path(args.output)
            result = json.loads((output / "completed.json").read_text())
            plan = json.loads((output / "plan.json").read_text())
            planned_updates = len(plan["arms"][args.arm])
            self.assertEqual(result["termination"], "budget")
            self.assertEqual(result["state"]["update"], planned_updates)
            self.assertEqual(result["state"]["plan_cursor"]["cursor"], planned_updates)
            self.assertEqual(set(result["state"]["stage_histogram"]), {"0", "1", "2", "3"})
            self.assertGreaterEqual(result["state"]["compute_units"], args.budget)
            events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            updates = [event for event in events if event["event"] == "update"]
            self.assertEqual(len(updates), planned_updates)
            self.assertGreater(len(updates), 0)
            for event in updates:
                self.assertEqual(event["loss"], 0.0)
                self.assertEqual(event["grad_norm"], 0.0)
            payload = torch.load(Path(result["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["cuda_rng"], [])
            self.assertEqual(len(payload["optimizer"]["state"]), 1)
            for state in payload["optimizer"]["state"].values():
                self.assertEqual(state["step"].item(), planned_updates)
                for key in ("exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.isfinite(state[key]).all())
                    self.assertEqual(torch.count_nonzero(state[key]).item(), 0)
            self.assertTrue(torch.isfinite(model.anchor).all())

    def test_cpu_dropout_resume_restores_plan_optimizer_rng_and_rejects_foreign_state(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            stack.callback(torch.set_num_threads, previous_threads)
            # CPU training must never initialize or collect a CUDA RNG stream,
            # even when this test is run on the training host with GPUs visible.
            stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU test initialized CUDA")))
            stack.enter_context(patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CPU checkpoint read CUDA RNG")))
            stack.enter_context(patch.object(torch.cuda, "set_rng_state_all", side_effect=AssertionError("CPU resume restored CUDA RNG")))
            stack.enter_context(patch.object(torch.cuda, "manual_seed_all"))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            stack.enter_context(patch.object(trainer, "evaluate", return_value={"metrics": {}, "count": 6}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

            rows = [
                {"id": f"v3-row-{index}", "prompt": f"example{index}:",
                 "answer": "ABCDEFGH"[index], "family": "pointer_chasing", "difficulty": difficulty}
                for index, difficulty in enumerate((1, 2, 3, 4, 6, 8))
            ]
            data = root / "data"
            data.mkdir()
            for split in ("train", "dev"):
                (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

            seed_process(802)
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
                return OuroDepthModel(copy.deepcopy(initial), mode="full", lora_rank=32, checkpointing=True)

            initializer = fresh_model().save_trainable(root / "init")
            self.assertTrue(initializer.is_file())

            def arguments(name):
                return SimpleNamespace(
                    output=str(root / name), data_dir=str(data), model_path=str(root / "same-base"),
                    checkpoint=str(initializer), arm="independent", seed=73,
                    device="cpu", mode="full", lora_rank=32,
                    batch_size=2, micro_batch=1, eval_batch=2,
                    lr=1e-3, weight_decay=0.01, clip=1.0, warmup_fraction=0.05,
                    budget=4096, max_updates=100, max_length=16, padding_width=0,
                    train_limit=0, dev_limit=0, pad_id=0, eval_every=0,
                    save_every=2, depths=[4, 8], resume=None, plan_path=None,
                )

            original_collate = trainer.collate_fixed

            def trace_rng(trace):
                def collate_with_probes(*args, **kwargs):
                    # The frozen plan itself needs no runtime Python/NumPy
                    # draws. Probes make restoring those streams observable.
                    trace.append((random.random(), float(np.random.random())))
                    return original_collate(*args, **kwargs)
                return patch.object(trainer, "collate_fixed", side_effect=collate_with_probes)

            continuous_args = arguments("continuous")
            continuous = fresh_model()
            continuous_probes = []
            seed_process(continuous_args.seed)
            with trace_rng(continuous_probes):
                trainer.train(continuous, _TinyTokenizer(), continuous_args)
            continuous_rng = process_rng()
            continuous_output = Path(continuous_args.output)
            continuous_final = json.loads((continuous_output / "completed.json").read_text())
            self.assertEqual(continuous_final["termination"], "budget")

            interrupted_args = arguments("interrupted")
            interrupted = fresh_model()
            interrupted_probes = []
            original_checkpoint = trainer.checkpoint

            def save_then_interrupt(model, optimizer, output_dir, state, identity):
                result = original_checkpoint(model, optimizer, output_dir, state, identity)
                if state["update"] == 2:
                    raise _InterruptedAfterTwo()
                return result

            seed_process(interrupted_args.seed)
            with trace_rng(interrupted_probes), patch.object(trainer, "checkpoint", side_effect=save_then_interrupt):
                with self.assertRaises(_InterruptedAfterTwo):
                    trainer.train(interrupted, _TinyTokenizer(), interrupted_args)

            output = Path(interrupted_args.output)
            source = output / "checkpoint-2"
            training_path = source / "training.pt"
            saved_at_two = torch.load(training_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved_at_two["state"]["update"], 2)
            self.assertEqual(saved_at_two["state"]["plan_cursor"]["cursor"], 2)
            self.assertEqual(saved_at_two["cuda_rng"], [])
            self.assertFalse((output / "completed.json").exists())
            for value in saved_at_two["optimizer"]["state"].values():
                self.assertEqual(value["step"].item(), 2)
                self.assertTrue(torch.isfinite(value["exp_avg"]).all())
                self.assertTrue(torch.isfinite(value["exp_avg_sq"]).all())

            plan_path = output / "plan.json"
            plan_text = plan_path.read_text()
            plan = json.loads(plan_text)
            self.assertEqual(plan, json.loads((continuous_output / "plan.json").read_text()))
            self.assertEqual(plan["padding_width"], 8)
            self.assertEqual(plan["num_layers"], 1)
            self.assertEqual(saved_at_two["state"]["plan_cursor"]["plan_fingerprint"], plan["fingerprint"])
            identity_path = output / "identity.json"
            identity_text = identity_path.read_text()
            metrics_text = (output / "metrics.jsonl").read_text()
            resume_args = copy.copy(interrupted_args)
            resume_args.resume = str(source)

            wrong_arm = copy.copy(resume_args)
            wrong_arm.arm = "conditional"
            with self.assertRaises(ValueError):
                trainer.train(fresh_model(), _TinyTokenizer(), wrong_arm)

            foreign = root / "foreign" / "checkpoint-2"
            shutil.copytree(source, foreign)
            wrong_source = copy.copy(resume_args)
            wrong_source.resume = str(foreign)
            with self.assertRaises(ValueError):
                trainer.train(fresh_model(), _TinyTokenizer(), wrong_source)

            mutated_plan = copy.deepcopy(plan)
            mutated_plan["arms"]["independent"][0]["depth"] += 2
            try:
                plan_path.write_text(json.dumps(mutated_plan))
                with self.assertRaises(ValueError):
                    trainer.train(fresh_model(), _TinyTokenizer(), resume_args)
            finally:
                plan_path.write_text(plan_text)

            original_payload = training_path.read_bytes()
            for corruption in ("checkpoint_identity", "prefix_compute", "cursor_update_disagreement"):
                with self.subTest(corruption=corruption):
                    payload = copy.deepcopy(saved_at_two)
                    if corruption == "checkpoint_identity":
                        payload["identity"]["seed"] += 1
                    elif corruption == "prefix_compute":
                        payload["state"]["compute_units"] += 1
                    else:
                        payload["state"]["plan_cursor"]["cursor"] += 1
                    try:
                        torch.save(payload, training_path)
                        with self.assertRaises(ValueError):
                            trainer.train(fresh_model(), _TinyTokenizer(), resume_args)
                    finally:
                        training_path.write_bytes(original_payload)
            checkpoint_identity_path = source / "identity.json"
            checkpoint_identity_text = checkpoint_identity_path.read_text()
            checkpoint_identity = json.loads(checkpoint_identity_text)
            checkpoint_identity["seed"] += 1
            try:
                checkpoint_identity_path.write_text(json.dumps(checkpoint_identity))
                with self.assertRaises(ValueError):
                    trainer.train(fresh_model(), _TinyTokenizer(), resume_args)
            finally:
                checkpoint_identity_path.write_text(checkpoint_identity_text)
            self.assertEqual(identity_path.read_text(), identity_text)
            self.assertEqual(plan_path.read_text(), plan_text)
            self.assertEqual((output / "metrics.jsonl").read_text(), metrics_text)
            self.assertFalse((output / "completed.json").exists())

            resumed = fresh_model()
            resumed_probes = []
            seed_process(9999)
            random.random()
            np.random.rand(19)
            torch.randn(97)  # Resume must override all three unrelated RNG streams.
            with trace_rng(resumed_probes):
                trainer.train(resumed, _TinyTokenizer(), resume_args)
            self.assertEqual(continuous_probes, interrupted_probes + resumed_probes)
            self.assert_tree_equal(continuous_rng, process_rng())
            self.assert_tree_equal(continuous.state_dict(), resumed.state_dict())

            resumed_final = json.loads((output / "completed.json").read_text())
            self.assertEqual(resumed_final["termination"], "budget")
            self.assert_tree_equal(continuous_final["state"], resumed_final["state"])
            self.assertEqual(identity_path.read_text(), identity_text)
            self.assertEqual(plan_path.read_text(), plan_text)
            records = plan["arms"]["independent"]
            self.assertEqual(len(continuous_probes), len(records) * 2)
            state = resumed_final["state"]
            self.assertEqual(state["update"], len(records))
            self.assertGreater(state["update"], 2)
            self.assertEqual(state["plan_cursor"]["cursor"], len(records))
            self.assertEqual(state["compute_units"], sum(row["compute_units"] for row in records))
            self.assertGreaterEqual(state["compute_units"], interrupted_args.budget)
            self.assertEqual(state["examples"], len(records) * 2)
            self.assertEqual(state["padded_tokens"], len(records) * 2 * 8)
            self.assertEqual(state["valid_tokens"], sum(
                len(_TinyTokenizer().encode(rows[index]["prompt"]))
                for record in records for index in record["indices"]
            ))
            for name, field in (("depth_histogram", "depth"), ("task_histogram", "difficulty"), ("stage_histogram", "stage")):
                prefix = "pointer_chasing/d" if field == "difficulty" else ""
                self.assertEqual(state[name], dict(Counter(
                    prefix + str(record[field]) for record in records for _ in record["indices"]
                )))
            self.assertEqual(set(state["stage_histogram"]), {"0", "1", "2", "3"})

            continuous_checkpoint = torch.load(
                Path(continuous_final["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False
            )
            resumed_checkpoint = torch.load(
                Path(resumed_final["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False
            )
            for key in ("optimizer", "state", "torch_rng", "cuda_rng", "python_rng", "numpy_rng"):
                self.assert_tree_equal(continuous_checkpoint[key], resumed_checkpoint[key])
            self.assertEqual(resumed_checkpoint["identity"], saved_at_two["identity"])
            self.assertEqual(resumed_checkpoint["cuda_rng"], [])
            self.assertTrue(any(
                not torch.equal(saved_at_two["optimizer"]["state"][key]["exp_avg"], value["exp_avg"])
                for key, value in resumed_checkpoint["optimizer"]["state"].items()
            ))
            for value in resumed_checkpoint["optimizer"]["state"].values():
                self.assertEqual(value["step"].item(), len(records))
                self.assertTrue(torch.isfinite(value["exp_avg"]).all())
                self.assertTrue(torch.isfinite(value["exp_avg_sq"]).all())

            def update_trace(path):
                trace = [json.loads(line) for line in path.read_text().splitlines()]
                return [
                    {key: event[key] for key in ("update", "depth", "loss", "grad_norm", "lr", "compute_units")}
                    for event in trace if event["event"] == "update"
                ]

            continuous_trace = update_trace(continuous_output / "metrics.jsonl")
            resumed_trace = update_trace(output / "metrics.jsonl")
            self.assertEqual(continuous_trace, resumed_trace)
            self.assertEqual([event["update"] for event in resumed_trace], list(range(1, len(records) + 1)))
            cumulative_compute = 0
            for event, record in zip(resumed_trace, records):
                cumulative_compute += record["compute_units"]
                self.assertEqual(event["depth"], record["depth"])
                self.assertEqual(event["compute_units"], cumulative_compute)
                self.assertEqual(event["lr"], interrupted_args.lr * lr_multiplier(record["lr_progress"], interrupted_args.warmup_fraction))
                self.assertTrue(math.isfinite(event["loss"]))
                self.assertTrue(math.isfinite(event["grad_norm"]) and event["grad_norm"] > 0)


if __name__ == "__main__":
    unittest.main()
