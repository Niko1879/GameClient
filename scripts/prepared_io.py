"""Canonical JSON, file digests and memory accounting shared by preparation and training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

# Terrain height, cliffs, public supply sources and cell coordinates do not
# change within a perspective; prepared events share one copy of them.
STATIC_MAP_COLUMNS = (0, 1, 9, 10, 11)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


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
    """Bytes of a row's encoded feature arrays: what a chunk stores, without its transient entity dicts."""
    return sum(value.nbytes for value in encoded.values() if isinstance(value, np.ndarray))


def _common(row):
    """Perspective-constant inputs stored once per prepared episode."""
    encoded = row['encoded']
    shared = {key: np.array(encoded[key], copy=True, order='K') for key in ('low', 'span') if key in encoded}
    maps = encoded.get('map_inputs')
    if isinstance(maps, np.ndarray) and maps.ndim == 2 and maps.shape[1] > max(STATIC_MAP_COLUMNS):
        shared.update(map_static=np.array(maps[:, STATIC_MAP_COLUMNS], copy=True, order='C'),
                      map_shape=list(maps.shape))
    return shared
