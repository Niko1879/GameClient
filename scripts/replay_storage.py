"""Native training export containers: block-compressed binary or readable JSONL."""
from __future__ import annotations

from pathlib import Path


def open_trajectory(path, mode='rt'):
    return Path(path).open(mode, **({'encoding': 'utf-8'} if 't' in mode else {}))


def manifest_path(path):
    return Path(path).with_suffix('.manifest.json')


def verify_storage(path, manifest):
    storage = manifest.get('storage')
    if storage is None:
        if Path(path).suffix.lower() != '.jsonl':
            raise ValueError('Trajectory lacks storage provenance')
        return
    from binary_replay import MAGIC
    if storage.get('codec') != 'gobs1' or Path(path).suffix.lower() != '.bin':
        raise ValueError('Unsupported replay storage provenance')
    with Path(path).open('rb') as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError('Unsupported binary replay signature/version')
    if storage['stored_bytes'] != Path(path).stat().st_size:
        raise ValueError('Binary replay stored size changed')
    # Ordinary full-source SHA checks bind the blocks; audit() separately
    # checks every block CRC, typed value and observation invariant.
