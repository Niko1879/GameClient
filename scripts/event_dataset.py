"""Prepared event datasets: verified shard reads and the deterministic training sampler.

Reading and sampling never change prepared data, so this module is outside the
feature identity (event_store.FEATURE_SOURCES) and can change without invalidating
the corpus.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from event_store import SCHEMA
from generals_policy import ARCHITECTURE
from prepared_io import COMMON_FIELDS, _digest, _json_bytes
from tensor_replay_store import ARRAY_FIELDS, LIST_FIELDS, MASK_FIELDS, decode_columns


class _Views:
    """A decoded column's ragged index as Python lists: one conversion per chunk."""
    __slots__ = ('data', 'ids', 'offsets', 'shapes', 'orders')

    def __init__(self, column):
        self.data, self.ids, self.offsets = column.data, column.ids.tolist(), column.offsets.tolist()
        self.shapes = [tuple(shape) for shape in column.shapes.tolist()]
        self.orders = ['F' if order else 'C' for order in column.orders.tolist()]

    def at(self, row):
        index = self.ids[row]
        if index < 0:
            return None
        return self.data[self.offsets[index]:self.offsets[index+1]].reshape(self.shapes[index], order=self.orders[index])


def chunk_rows(columns, metadata, common, key):
    """Every row of a decoded chunk, equal to tensor_replay_store.TensorRows' views.

    TensorRows is a feature source (editing it changes the feature key), so the
    training reader builds the same rows here in one pass over the chunk.
    test_event_dataset pins the two to identical rows on stored shards.
    """
    views = {name: _Views(column) for name, column in columns.items()}
    encoded_fields = [(name, views['e/'+name]) for name in ARRAY_FIELDS if 'e/'+name in views]
    mask_fields = [(name, views['l/'+name]) for name in MASK_FIELDS if 'l/'+name in views]
    list_fields = [(name, views['l/'+name]) for name in LIST_FIELDS if 'l/'+name in views]
    variants = {}
    for name in columns:
        if '|' in name:
            variants.setdefault(name.partition('|')[0], []).append(views[name])
    result = []
    for item, meta in enumerate(metadata):
        encoded = {}
        for name, view in encoded_fields:
            value = view.at(item)
            if value is not None:
                encoded[name] = value
        chosen = {}
        for name, candidates in variants.items():
            values = [value for view in candidates if (value := view.at(item)) is not None]
            if len(values) > 1:
                raise ValueError('Overlapping training column variants')
            if values:
                chosen[name] = values[0]
        encoded.update({name[2:]: value for name, value in chosen.items() if name.startswith('e/')})
        encoded.update(meta['encoded'])
        for name in COMMON_FIELDS:
            if name not in encoded and name in common:
                encoded[name] = common[name]
        target = meta['row']['target']
        mask = {'key': key, 'teacher': target}
        mask.update({name[2:]: value for name, value in chosen.items() if name.startswith('l/')})
        for name, view in mask_fields:
            value = view.at(item)
            if value is not None:
                mask[name] = value
        for name, view in list_fields:
            value = view.at(item)
            if value is not None:
                mask[name] = value.tolist()
        encoded['_training_legality'] = mask
        result.append({**meta['row'], 'encoded': encoded})
    return result


class EventDataset:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.digest = _digest(self.path)
        self.document = json.loads(self.path.read_text())
        if (self.document.get('kind') != 'generals_event_dataset' or self.document.get('schema') != SCHEMA
                or self.document.get('complete') is False):
            raise ValueError('Unsupported event dataset')
        self.config = self.document['config']
        if self.config.get('architecture') != ARCHITECTURE:
            raise ValueError('Not an event policy dataset')
        seen, keys = set(), set()
        for entry in self.document['entries']:
            if hashlib.sha256(_json_bytes(entry['index'])).hexdigest() != entry['index_sha256']:
                raise ValueError('Event dataset index proof mismatch')
            match = entry['match_key']
            if match in seen or entry['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Duplicate match or invalid whole-match split')
            seen.add(match); keys.add(entry['index']['legality_key'])
        if len(keys) != 1:
            raise ValueError('Event dataset mixes feature identities')
        self.key = next(iter(keys))
        self.entries = self.document['entries']
        self.verified, self.shared = {}, {}

    def shard_path(self, entry):
        path = (self.path.parent / entry['path']).resolve()
        if self.path.parent not in path.parents:
            raise ValueError('Event shard path escapes dataset directory')
        return path

    def chunks(self, split):
        for source, entry in enumerate(self.entries):
            if entry['split'] != split:
                continue
            for episode in entry['index']['episodes']:
                for descriptor in episode['chunks']:
                    yield source, episode['shared'], descriptor

    def read(self, reference):
        source, shared, descriptor = reference
        entry = self.entries[source]
        path = self.shard_path(entry)
        stamp = path.stat().st_size, path.stat().st_mtime_ns
        if source not in self.verified:
            if _digest(path) != entry['archive_sha256']:
                raise ValueError('Event shard integrity mismatch')
            self.verified[source] = stamp
        elif self.verified[source] != stamp:
            raise ValueError('Event shard changed while training')

        def read_member(archive, item):
            if archive.getinfo(item['member']).file_size != item['bytes']:
                raise ValueError('Event column size mismatch')
            raw = archive.read(item['member'])
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Event column hash mismatch')
            return decode_columns(raw)

        with zipfile.ZipFile(path) as archive:
            identity = source, shared
            if identity not in self.shared:
                columns, meta = read_member(archive, entry['index']['shared'][shared])
                common = {name: column.at(0) for name, column in columns.items()}
                common.update(meta[0])
                self.shared = {identity: common}  # One immutable map working set.
            columns, metadata = read_member(archive, descriptor)
        if len(metadata) != descriptor['rows']:
            raise ValueError('Event column row count mismatch')
        rows = chunk_rows(columns, metadata, self.shared[identity], self.key)
        # Not stored in the shard: run_prepared_training.selection adds it from the replay.
        enemy = tuple(entry.get('enemy_identity', (0, 0)))
        for row in rows:
            row['encoded']['enemy_identity'] = enemy
        return rows

    def windows(self, split, *, seed, epoch, buffer_bytes=64*1024*1024):
        """Shuffled chunk references grouped into byte-bounded windows.

        Windows are independent: each has its own deterministic RNG, so readers
        may decode different windows concurrently without changing any batch.
        """
        if buffer_bytes < 1:
            raise ValueError('Positive shuffle-buffer budget required')
        chunks = list(self.chunks(split))
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(chunks))
        windows, current, size = [], [], 0
        for ordinal in order:
            reference = chunks[int(ordinal)]
            estimate = reference[2]['bytes']*3
            if current and size+estimate > buffer_bytes:
                windows.append(current); current, size = [], 0
            current.append(reference); size += estimate
        if current:
            windows.append(current)
        return windows

    def window_batches(self, references, *, seed, epoch, window, batch_size=256, start=0):
        """One window's batches, as (rows, position after this batch), from `start`."""
        if batch_size < 1:
            raise ValueError('Positive batch budget required')
        rows = []
        for reference in references:
            rows.extend(self.read(reference))
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, window, 1]))
        rng.shuffle(rows)
        # Stable sorting keeps random order inside entity-count buckets.
        rows.sort(key=lambda row: len(row['encoded']['numeric'])//16)
        batches = [rows[first:first+batch_size] for first in range(0, len(rows), batch_size)]
        positions = rng.permutation(len(batches))
        for position in range(start, len(positions)):
            yield batches[int(positions[position])], position+1

    def batches(self, split, *, seed, epoch, batch_size=256, buffer_bytes=64*1024*1024,
                cursor=(0, 0)):
        """Shuffle chunks, then bucket a byte-bounded window by entity count.

        Cursor is (window, batch). Rebuilding it needs neither earlier decoded
        data nor neural state. All windows use their own deterministic RNG.
        """
        if batch_size < 1 or buffer_bytes < 1:
            raise ValueError('Positive batch and shuffle-buffer budgets required')
        windows = self.windows(split, seed=seed, epoch=epoch, buffer_bytes=buffer_bytes)
        for window in range(cursor[0], len(windows)):
            start = cursor[1] if window == cursor[0] else 0
            for rows, position in self.window_batches(windows[window], seed=seed, epoch=epoch, window=window,
                                                      batch_size=batch_size, start=start):
                yield rows, (window, position)
