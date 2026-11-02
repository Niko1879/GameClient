"""Canonical JSON, file digests and memory accounting shared by preparation and training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

# Extent, terrain, public supply sources and cell coordinates do not change within a
# perspective; prepared events share one copy of them.
COMMON_FIELDS = ('low', 'span', 'map_static')


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def _digest(path):
    from replay_training import file_sha256
    return file_sha256(path)


def _dtype(value):
    dtype = np.dtype(value)
    if dtype.hasobject or dtype.fields or dtype.kind not in 'biufcSU' or dtype.itemsize < 1:
        raise ValueError('Unsupported prepared array dtype: ' + str(dtype))
    return dtype


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.write-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _array_bytes(encoded):
    """Bytes of a row's encoded feature arrays: what a chunk stores, without its transient entity dicts
    or the per-episode inputs in COMMON_FIELDS."""
    return sum(value.nbytes for name, value in encoded.items() if isinstance(value, np.ndarray) and name not in COMMON_FIELDS)


def _common(row):
    """Perspective-constant inputs stored once per prepared episode."""
    encoded = row['encoded']
    return {key: np.array(encoded[key], copy=True, order='K') for key in COMMON_FIELDS if key in encoded}
