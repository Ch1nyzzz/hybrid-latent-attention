"""New model-only transfer boundary, using small tied Torch modules on CPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from ouro_depth.huginn_initialization import load_common_weights
from ouro_depth.huginn_training import _model_metadata


class Config:
    def to_dict(self):
        return {'tie_embeddings': True, 'test_time_noise': 0, 'fixture': 'CPU_synthetic_only'}


class Tiny(torch.nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        self.config = Config()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)
        if tied:
            self.head.weight = self.embedding.weight
        self.core = torch.nn.Linear(4, 4)
        self.register_buffer('position', torch.arange(4, dtype=torch.float32))


def saved(path, model):
    identity = {'run': {'scope': 'synthetic_qualified_common'}, **_model_metadata(model)}
    payload = {'format_version': 1,
               'parameters': {k: v.detach().clone() for k, v in model.named_parameters()},
               'buffers': {k: v.detach().clone() for k, v in model.named_buffers()}}
    torch.save(payload, path / 'model.pt')
    # Deliberately unreadable: a model-only transfer must never parse training.pt.
    (path / 'training.pt').write_bytes(b'not an optimizer or RNG checkpoint')
    (path / 'identity.json').write_text(json.dumps(identity))
    (path / 'complete.json').write_text(json.dumps({'format_version': 1, 'update': 256,
        'files': ['model.pt', 'training.pt', 'identity.json']}))
    return identity, payload


def stat(path):
    value = (path / 'model.pt').stat()
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns}


class ModelOnlyInitialization(unittest.TestCase):
    def test_exact_weights_buffers_tie_and_fresh_optimizer_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); source = Tiny(); target = Tiny()
            identity, payload = saved(path, source)
            optimizer = torch.optim.AdamW(target.parameters(), lr=1e-6)
            rng = torch.get_rng_state().clone()
            receipt = load_common_weights(target, path, expected_identity=identity, expected_file_stat=stat(path))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(optimizer.state, {})
            self.assertIs(target.embedding.weight, target.head.weight)
            for name, tensor in target.named_parameters():
                self.assertTrue(torch.equal(tensor, payload['parameters'][name]))
            for name, tensor in target.named_buffers():
                self.assertTrue(torch.equal(tensor, payload['buffers'][name]))
            self.assertFalse(receipt['optimizer_state_loaded']); self.assertFalse(receipt['training_rng_loaded'])

    def test_malformed_last_tensor_identity_stat_alias_and_nonfinal_fail_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); source = Tiny(); target = Tiny()
            identity, original = saved(path, source)
            unchanged = {k: v.detach().clone() for k, v in target.state_dict().items()}
            bad = copy.deepcopy(original); bad['buffers']['position'][-1] = float('nan')
            torch.save(bad, path / 'model.pt')
            with self.assertRaisesRegex(ValueError, 'Invalid common checkpoint tensor'):
                load_common_weights(target, path, expected_identity=identity, expected_file_stat=stat(path))
            self.assertTrue(all(torch.equal(v, unchanged[k]) for k, v in target.state_dict().items()))
            identity, _ = saved(path, source)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                load_common_weights(target, path, expected_identity={**identity, 'run': {'different': True}}, expected_file_stat=stat(path))
            with self.assertRaisesRegex(ValueError, 'file changed'):
                load_common_weights(target, path, expected_identity=identity, expected_file_stat={**stat(path), 'size': 1})
            with self.assertRaisesRegex(ValueError, 'alias graph differs'):
                load_common_weights(Tiny(tied=False), path, expected_identity=identity, expected_file_stat=stat(path))
            marker = json.loads((path / 'complete.json').read_text()); marker['update'] = 128
            (path / 'complete.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, 'final256'):
                load_common_weights(target, path, expected_identity=identity, expected_file_stat=stat(path))


if __name__ == '__main__':
    unittest.main()
