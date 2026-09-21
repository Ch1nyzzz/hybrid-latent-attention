"""Explicit provenance for fresh Stage1 initialization on a new OPD corpus."""
import re


def validate_stage1_dataset(payload, target_manifest, expected_source=None):
    meta = payload.get('metadata', {})
    if meta.get('stage') != 1 or payload.get('step') != meta.get('steps'):
        raise ValueError('Direct decode training requires a completed Stage1 export')
    source = meta.get('data_manifest_sha256')
    for digest in (source, target_manifest):
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid dataset manifest digest')
    if expected_source is not None and source != expected_source:
        raise ValueError('Stage1 source manifest does not match explicit expected manifest')
    if source != target_manifest and expected_source is None:
        raise ValueError('Stage1 and decode training corpus manifests differ')
