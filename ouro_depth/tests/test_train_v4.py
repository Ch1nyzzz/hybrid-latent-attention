"""Actual tiny Ouro CPU V4 persistence, next-update and source-binding check.

Synthetic train/DEV only. This does not repeat the wrapper equivalence suite.
"""
from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ouro_depth import train_v4 as trainer
from ouro_depth.model import OuroDepthModel
from ouro_depth.vendor.configuration_ouro import OuroConfig
from ouro_depth.vendor.modeling_ouro import OuroForCausalLM


class TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" "):
            return [2 + "ABCDEFGH".index(text[1:])]
        prompt, separator, answer = text.partition(" ")
        index = int(prompt.removeprefix("example").removesuffix(":"))
        ids = [1, 11 + index, 20 + index] + [31] * (index % 3)
        return ids + ([2 + "ABCDEFGH".index(answer)] if separator else [])


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state().clone()}


class Interrupted(Exception):
    pass


class V4TrainerCPU(unittest.TestCase):
    def equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertIs(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_source_receipt_is_relocatable_but_detects_changed_execution_dependency(self):
        receipt = trainer.source_receipt()
        package = Path(trainer.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            mirror = Path(temporary)
            for name in receipt["files"]:
                target = mirror / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package / name, target)
            self.assertEqual(trainer.source_receipt(mirror), receipt)
            changed = mirror / "train_v4.py"
            changed.write_text(changed.read_text() + "\n# altered frozen source\n")
            self.assertNotEqual(trainer.source_receipt(mirror), receipt)

    def test_dropout_full_bptt_resume_matches_next_updates_adam_rng_and_rejects_bad_state(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            stack.callback(torch.set_num_threads, previous_threads)
            stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU check initialized CUDA")))
            stack.enter_context(patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CPU checkpoint queried CUDA RNG")))
            stack.enter_context(patch.object(torch.cuda, "set_rng_state_all", side_effect=AssertionError("CPU resume restored CUDA RNG")))
            stack.enter_context(patch.object(torch.cuda, "manual_seed_all"))
            stack.enter_context(patch.object(trainer, "evaluate", return_value={"count": 24, "metrics": {}}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            rows = [{"id": f"v4-synthetic-{i}", "prompt": f"example{i}:", "answer": "ABCDEFGH"[i % 8],
                     "family": "pointer_chasing", "difficulty": (1, 2, 3, 4, 6, 8)[i // 4]} for i in range(24)]
            data = root / "data"
            data.mkdir()
            for name in ("train.jsonl", "dev.jsonl"):
                (data / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
            seed(802)
            config = OuroConfig(vocab_size=80, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=32, attention_dropout=0.1, pad_token_id=0,
                bos_token_id=1, eos_token_id=2, use_cache=False, tie_word_embeddings=False)
            config._attn_implementation = "sdpa"
            initial = OuroForCausalLM(config).float()

            def fresh():
                return OuroDepthModel(copy.deepcopy(initial), mode="full", checkpointing=True)

            initializer = fresh().save_trainable(root / "initializer")
            plan_path = root / "frozen-plan.json"

            def args(name):
                return SimpleNamespace(output=str(root / name), data_dir=str(data),
                    model_path=str(root / "same-base"), checkpoint=str(initializer),
                    plan_path=str(plan_path), arm="fixed8", seed=20260915, device="cpu",
                    mode="full", lora_rank=32, batch_size=2, micro_batch=1,
                    fixed4_updates=12, max_length=16, padding_width=8, max_updates=12,
                    lr=1e-5, weight_decay=0.01, clip=1.0, train_limit=0, dev_limit=0,
                    eval_batch=2, eval_every=2, save_every=2, depths=[4, 8, 16],
                    pad_id=0, resume=None)

            setup = args("unused")
            setup.plan_path = None
            plan, _, _ = trainer.prepare_plan(TinyTokenizer(), setup, 1)
            plan_path.write_text(json.dumps(plan))
            original_collate = trainer.collate_fixed

            def trace_rng(trace):
                def collate(*a, **kw):
                    trace.append((random.random(), float(np.random.random())))
                    return original_collate(*a, **kw)
                return patch.object(trainer, "collate_fixed", side_effect=collate)

            continuous_args, continuous, continuous_trace = args("continuous"), fresh(), []
            seed(continuous_args.seed)
            with trace_rng(continuous_trace):
                continuous_result = trainer.train(continuous, TinyTokenizer(), continuous_args)
            continuous_rng = rng()
            self.assertEqual(continuous_result["termination"], "budget")
            self.assertEqual(continuous_result["state"]["update"], 6)
            continuous_saved = torch.load(Path(continuous_result["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False)
            interrupted_args, interrupted, interrupted_trace = args("interrupted"), fresh(), []
            original_checkpoint = trainer.checkpoint

            def checkpoint_then_interrupt(model, optimizer, output, state, identity):
                result = original_checkpoint(model, optimizer, output, state, identity)
                if state["update"] == 2:
                    raise Interrupted()
                return result

            seed(interrupted_args.seed)
            with trace_rng(interrupted_trace), patch.object(trainer, "checkpoint", side_effect=checkpoint_then_interrupt):
                with self.assertRaises(Interrupted):
                    trainer.train(interrupted, TinyTokenizer(), interrupted_args)
            output = Path(interrupted_args.output)
            saved_path = output / "checkpoint-2"
            training_path = saved_path / "training.pt"
            before_saved_bytes = training_path.read_bytes()
            at_two = torch.load(training_path, map_location="cpu", weights_only=False)
            self.assertEqual(at_two["state"]["update"], 2)
            self.assertEqual(at_two["cuda_rng"], [])
            resume_args = copy.copy(interrupted_args)
            resume_args.resume = str(saved_path)
            wrong_arm = copy.copy(resume_args)
            wrong_arm.arm = "fixed4"
            wrong_arm.depths = [4, 6, 8, 16]
            with self.assertRaises(ValueError):
                trainer.train(fresh(), TinyTokenizer(), wrong_arm)
            foreign = root / "foreign" / "checkpoint-2"
            shutil.copytree(saved_path, foreign)
            wrong_source = copy.copy(resume_args)
            wrong_source.resume = str(foreign)
            with self.assertRaises(ValueError):
                trainer.train(fresh(), TinyTokenizer(), wrong_source)
            local_plan = output / "plan.json"
            plan_text = local_plan.read_text()
            try:
                altered = json.loads(plan_text)
                altered["arms"]["fixed8"][0]["depth"] = 4
                local_plan.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    trainer.train(fresh(), TinyTokenizer(), resume_args)
            finally:
                local_plan.write_text(plan_text)
            for fault in ("identity", "compute", "cursor", "adam_step"):
                payload = copy.deepcopy(at_two)
                if fault == "identity": payload["identity"]["seed"] += 1
                elif fault == "compute": payload["state"]["compute_units"] += 1
                elif fault == "cursor": payload["state"]["plan_cursor"]["cursor"] += 1
                elif fault == "adam_step": next(iter(payload["optimizer"]["state"].values()))["step"] += 1
                try:
                    torch.save(payload, training_path)
                    with self.subTest(fault=fault), self.assertRaises(ValueError):
                        trainer.train(fresh(), TinyTokenizer(), resume_args)
                finally:
                    training_path.write_bytes(before_saved_bytes)
            frozen_code = output / "source" / "ouro_depth" / "train_v4.py"
            frozen_bytes = frozen_code.read_bytes()
            try:
                frozen_code.write_bytes(frozen_bytes + b"\n# changed after checkpoint\n")
                with self.assertRaises(ValueError):
                    trainer.train(fresh(), TinyTokenizer(), resume_args)
            finally:
                frozen_code.write_bytes(frozen_bytes)
            original_metrics = (output / "metrics.jsonl").read_text()
            changed_receipt = copy.deepcopy(trainer.source_receipt())
            changed_receipt["fingerprint"] = "changed-runtime-source"
            with patch.object(trainer, "source_receipt", return_value=changed_receipt), self.assertRaises(ValueError):
                trainer.train(fresh(), TinyTokenizer(), resume_args)
            self.assertEqual((output / "metrics.jsonl").read_text(), original_metrics)

            resumed, resumed_trace = fresh(), []
            seed(999)
            random.random(); np.random.random(13); torch.randn(97)
            with trace_rng(resumed_trace):
                resumed_result = trainer.train(resumed, TinyTokenizer(), resume_args)
            self.equal(continuous.state_dict(), resumed.state_dict())
            self.equal(continuous_rng, rng())
            self.assertEqual(continuous_trace, interrupted_trace + resumed_trace)
            self.equal(continuous_result["state"], resumed_result["state"])
            self.assertEqual(resumed_result["state"]["compute_units"], plan["budget"])
            self.assertEqual(resumed_result["state"]["task_histogram"], {f"pointer_chasing/d{d}": 2 for d in (1, 2, 3, 4, 6, 8)})
            resumed_saved = torch.load(Path(resumed_result["checkpoint"]) / "training.pt", map_location="cpu", weights_only=False)
            for key in ("optimizer", "state", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
                self.equal(continuous_saved[key], resumed_saved[key])
            for values in resumed_saved["optimizer"]["state"].values():
                self.assertEqual(values["step"].item(), 6)

            def updates(directory):
                events = [json.loads(l) for l in (directory / "metrics.jsonl").read_text().splitlines()]
                keys = ("update", "depth", "difficulty", "loss", "grad_norm", "lr", "compute_units", "missing_grad_count")
                return [{key: event[key] for key in keys} for event in events if event["event"] == "update"]

            uninterrupted_updates = updates(Path(continuous_args.output))
            resumed_updates = updates(output)
            self.assertEqual(uninterrupted_updates, resumed_updates)
            self.assertEqual([event["update"] for event in resumed_updates], list(range(1, 7)))
            for event in resumed_updates:
                self.assertEqual(event["lr"], 1e-5)
                self.assertEqual(event["depth"], 8)
                self.assertEqual(event["missing_grad_count"], 0)
                self.assertGreater(event["grad_norm"], 0)

            # A caller's engineering cap cannot certify the full budget.
            limited_args = args("limited")
            limited_args.max_updates = 1
            seed(limited_args.seed)
            limited = trainer.train(fresh(), TinyTokenizer(), limited_args)
            self.assertEqual(limited["termination"], "max_updates")
            self.assertFalse((Path(limited_args.output) / "completed.json").exists())
            self.assertTrue((Path(limited_args.output) / "incomplete.json").is_file())


if __name__ == "__main__":
    unittest.main()
