"""Pure-stdlib boundary checks: importing this entry point does not import torch."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("huginn_engineering_smoke", Path(__file__).with_name("smoke_gpu.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeBoundaryTests(unittest.TestCase):
    def test_integer_indices_include_exact_last_element_without_overflow(self):
        for size in (1, 2, 63, 64, 65, 83_635_200, 2**24 + 1, 2**31 - 1, 2**53 + 1):
            with self.subTest(numel=size):
                indices = smoke.integer_sample_indices(size)
                self.assertEqual(len(indices), 64)
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], size - 1)
                self.assertEqual(indices, sorted(indices))
                self.assertTrue(all(type(index) is int and 0 <= index < size for index in indices))
        self.assertEqual(max(smoke.integer_sample_indices(83_635_200)), 83_635_199)
        for bad in (0, -1, True, 83_635_200.0):
            with self.assertRaises(ValueError):
                smoke.integer_sample_indices(bad)

    def test_attempt_label_is_safe_new_and_confined(self):
        with tempfile.TemporaryDirectory(prefix="huginn-smoke-path-test-") as temporary:
            directory = Path(temporary).resolve()
            destination = smoke._attempt_directory(directory, "parameter-index-fix")
            self.assertEqual(destination, directory / "attempts/parameter-index-fix")
            for bad in ("", ".", "..", "../escape", "a/b", "/tmp/escape", "x y", "x.y", "中文", "a" * 65):
                with self.assertRaises(ValueError):
                    smoke._attempt_directory(directory, bad)
            destination.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                smoke._attempt_directory(directory, "parameter-index-fix")


if __name__ == "__main__":
    unittest.main()
