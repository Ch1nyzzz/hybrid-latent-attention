"""CPU regression checks for the actual vLLM constructor and job failure paths."""
import contextlib
import io
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from hla.vllm_latent import compare
from hla.vllm_latent.patch_triton import (
    LEGACY_LAUNCH, LEGACY_TILE, OLD_TILE,
    patch_backend_source, patch_installation, remove_legacy_tuning,
)

FIXTURE = Path(__file__).with_name("fixtures") / "triton_builder_v026.py"
SOURCE = FIXTURE.read_text()
LOGGING = Path(__file__).resolve().parents[1] / "vllm_latent/job_logging.sh"


class MetadataBase:
    def __init__(self, spec, names, config, device):
        self.vllm_config = config


def build_metadata(source, width, kv_heads=1, graph=False):
    modes = NS(FULL=1, FULL_DECODE_ONLY=2, FULL_AND_PIECEWISE=3)
    env = {"_MetadataBase": MetadataBase, "CUDAGraphMode": modes,
           "torch": NS(empty=lambda shape, **_: NS(shape=tuple(shape)), float32="float32"),
           "get_num_attention_heads_from_layers": lambda *_: 16,
           "next_power_of_2": lambda n: 1 << (n - 1).bit_length(),
           "MIN_LAUNCH_GRID_SIZE_2D": 128, "NUM_PAR_SOFTMAX_SEGMENTS": 16}
    exec(compile(source, str(FIXTURE), "exec"), env)
    config = NS(model_config=NS(get_num_kv_heads=lambda _: 16, get_head_size=lambda: 128,
                               get_num_attention_heads=lambda _: 16, rswa_window=None),
                parallel_config=None,
                compilation_config=NS(cudagraph_mode=2 if graph else 0,
                                      cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]))
    spec = NS(block_size=16, num_kv_heads=kv_heads, head_size=width)
    return env["TritonAttentionMetadataBuilder"](spec, ["layer"], config, "cpu")


class TritonGeometryTests(unittest.TestCase):
    def test_original_eight_request_decode_writes_beyond_allocation(self):
        for width, ratio in [(256, 2), (512, 4)]:
            with self.subTest(width=width):
                md = build_metadata(SOURCE, width)
                self.assertEqual(md.seq_threshold_3D, 8)
                allocated = math.prod(md.softmax_segm_output.shape)
                # The kernel indexes [token, query_head, segment, actual head width].
                required = 8 * 16 * 16 * width
                self.assertEqual(required, ratio * allocated)
                self.assertGreater(required - 1, allocated - 1)

    def test_patched_groups_cover_all_decode_writes_in_eager_and_graph(self):
        fixed = patch_backend_source(SOURCE)
        for width, heads in [(512, 1), (256, 1), (128, 1), (128, 16)]:
            for graph in (False, True):
                with self.subTest(width=width, heads=heads, graph=graph):
                    md = build_metadata(fixed, width, heads, graph)
                    self.assertEqual(md.num_heads_kv, heads)
                    self.assertEqual(md.headdim, width)
                    self.assertEqual(md.seq_threshold_3D, 128 // heads)
                    for batch in (1, 2, 4, 8, 9, 32, 128):
                        if batch <= md.seq_threshold_3D:
                            self.assertLessEqual(batch * 16 * 16 * width,
                                                 math.prod(md.softmax_segm_output.shape))

    def test_original_ouro_geometry_is_preserved(self):
        before = build_metadata(SOURCE, 128, 16)
        after = build_metadata(patch_backend_source(SOURCE), 128, 16)
        self.assertEqual(before.softmax_segm_output.shape, after.softmax_segm_output.shape)
        self.assertEqual(before.seq_threshold_3D, after.seq_threshold_3D)

    def test_repeated_patch_and_unrecognized_source(self):
        fixed = patch_backend_source(SOURCE)
        self.assertEqual(patch_backend_source(fixed), fixed)
        with self.assertRaises(ValueError):
            patch_backend_source(SOURCE.replace("self.headdim = model_config.get_head_size()", "self.headdim = 256"))

    def test_legacy_tuning_is_undone_without_changing_other_kernels(self):
        kernel = "def tile(is_prefill):\n" + OLD_TILE + "    return 16\n\ndef launch():\n    launch_num_stages: int | None = None\n"
        legacy = kernel.replace(OLD_TILE, LEGACY_TILE) + LEGACY_LAUNCH
        self.assertEqual(remove_legacy_tuning(legacy), kernel)
        self.assertEqual(remove_legacy_tuning(kernel), kernel)
        with self.assertRaises(ValueError):
            remove_legacy_tuning(kernel.replace(OLD_TILE, LEGACY_TILE))

    def test_installation_rejects_partial_patch_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "v1/attention/backends/triton_attn.py"
            kernel = root / "v1/attention/ops/triton_unified_attention.py"
            backend.parent.mkdir(parents=True)
            kernel.parent.mkdir(parents=True)
            backend.write_text(SOURCE)
            kernel.write_text("# loop-scale patch: unknown version\n")
            with self.assertRaises(ValueError):
                patch_installation(root)
            self.assertEqual(backend.read_text(), SOURCE)
            kernel.write_text("# untouched kernel\n")
            self.assertEqual(len(patch_installation(root)), 1)
            self.assertEqual(patch_installation(root), [])
            self.assertEqual(backend.with_name(backend.name + ".loop-scale-before-cache-spec").read_text(), SOURCE)


class EvaluationHarnessTests(unittest.TestCase):
    def test_prompt_lengths_are_exact_for_short_and_long_inputs(self):
        tok = NS(encode=lambda s: list(range(len(s.split()))))
        for n in (1, 8, 16, 512, 4096):
            with self.subTest(n=n):
                self.assertEqual(len(compare.throughput_prompt_ids(tok, n)), n)
        self.assertGreater(len(compare.throughput_prompt_ids(tok, 0)), 0)
        with self.assertRaises(ValueError):
            compare.throughput_prompt_ids(tok, -1)

    def test_logged_failure_cannot_print_success_and_keeps_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "job.log"
            for code in (0, 23):
                result = subprocess.run(
                    ["bash", "-c", 'set -euo pipefail; source "$1"; run_logged "$2" bash -c "$3"; echo DONE',
                     "test", str(LOGGING), str(log), f"echo full-diagnostic >&2; exit {code}"],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, code)
                self.assertIn("full-diagnostic", log.read_text())
                self.assertEqual("DONE" in result.stdout, code == 0)

    def test_missing_reference_does_not_initialize_an_engine(self):
        with patch.object(sys, "argv", ["compare", "--model", "unused", "--out", "unused"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                compare.main()
        self.assertEqual(error.exception.code, 2)

    def test_reference_with_different_geometry_is_rejected_before_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.json"
            ref.write_text('{"student_cfg": {"rank": 512}, "prompts": [{}]}')
            argv = ["compare", "--model", "unused", "--student", "student", "--out", "unused", "--ref", str(ref)]
            with patch.object(sys, "argv", argv), patch.object(compare.torch, "load", return_value={"cfg": {"rank": 256}}):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    compare.main()
            self.assertEqual(error.exception.code, 2)

    def test_empty_reference_is_rejected_even_with_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.json"
            ref.write_text('{"student_cfg": {"rank": 512}, "prompts": []}')
            argv = ["compare", "--model", "unused", "--out", "unused", "--ref", str(ref), "--throughput", "8"]
            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    compare.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
