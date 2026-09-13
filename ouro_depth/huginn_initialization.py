"""Load a committed common model into a new experiment, without its Adam/RNG.

The caller must first qualify and freeze the final common-adaptation endpoint.
This helper validates that exact identity and live model schema before copying
any tensor. It does not select checkpoints, create optimizers, reseed training,
load training.pt, or authorize a new experiment. Own-run resume continues to use
huginn_training.load_checkpoint instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .huginn_training import FORMAT_VERSION, _json, _model_metadata


def load_common_weights(model, checkpoint, *, expected_identity, expected_file_stat):
    """Load final256 model tensors from an already qualified, frozen checkpoint.

    expected_file_stat is the caller's frozen model.pt size and mtime_ns. A full
    transfer/freeze digest may be recorded by the caller once; this helper does
    not repeatedly hash large checkpoints. Trusted directory ownership is part
    of that boundary. Every name, shape, dtype and finite value is checked before
    the first in-place copy. The embedding/head alias graph is preserved.
    """
    path = Path(checkpoint)
    if not isinstance(expected_identity, dict) or not expected_identity:
        raise ValueError('A qualified common-checkpoint identity is required')
    if (not isinstance(expected_file_stat, dict) or set(expected_file_stat) != {'size', 'mtime_ns'}
            or any(type(v) is not int or v <= 0 for v in expected_file_stat.values())):
        raise ValueError('Frozen model-file size and mtime_ns are required')
    marker = json.loads((path / 'complete.json').read_text())
    expected_marker = {'format_version': FORMAT_VERSION, 'update': 256,
                       'files': ['model.pt', 'training.pt', 'identity.json']}
    if marker != expected_marker or any(not (path / name).is_file() for name in expected_marker['files']):
        raise ValueError('A committed final256 common checkpoint is required')
    identity = json.loads((path / 'identity.json').read_text())
    if _json(identity) != _json(expected_identity):
        raise ValueError('Common checkpoint identity changed')
    metadata = _model_metadata(model)
    if any(_json(identity.get(key)) != _json(value) for key, value in metadata.items()):
        raise ValueError('Common model configuration, tensor schema or alias graph differs')
    weights_path = path / 'model.pt'
    before = weights_path.stat()
    if {'size': before.st_size, 'mtime_ns': before.st_mtime_ns} != expected_file_stat:
        raise ValueError('Frozen common-model file changed')
    payload = torch.load(weights_path, map_location='cpu', weights_only=True)
    after = weights_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Common-model file changed during loading')
    if (not isinstance(payload, dict) or set(payload) != {'format_version', 'parameters', 'buffers'}
            or payload['format_version'] != FORMAT_VERSION):
        raise ValueError('Unsupported common model payload')
    current = {'parameters': dict(model.named_parameters()), 'buffers': dict(model.named_buffers())}
    for field, tensors in current.items():
        saved = payload[field]
        if not isinstance(saved, dict) or saved.keys() != tensors.keys():
            raise ValueError(f'Common checkpoint {field} names differ')
        for name, value in saved.items():
            target = tensors[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f'Invalid common checkpoint tensor: {field}/{name}')
    with torch.no_grad():
        for field, tensors in current.items():
            for name, target in tensors.items():
                target.copy_(payload[field][name])
    return {'checkpoint': str(path), 'update': 256, 'model_file_stat': expected_file_stat.copy(),
            'parameter_tensors_loaded': len(current['parameters']),
            'parameter_elements_loaded': sum(p.numel() for p in current['parameters'].values()),
            'buffer_tensors_loaded': len(current['buffers']), 'all_saved_values_finite': True,
            'validated_alias_graph': metadata['model_aliases'], 'copy_method': 'in_place_preserving_aliases',
            'optimizer_state_loaded': False, 'training_rng_loaded': False,
            'scope': 'Model-only initialization; caller must construct fresh Adam and seed its new run'}
