"""Compressed training columns, with bounded decoding and no entity object tree.

Only frozen features, targets and teacher-prefix legality are stored.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import struct

import numpy as np

from prepared_io import COMMON_FIELDS, _dtype, _json_bytes

VERSION = 1
ARRAY_FIELDS = ('numeric', 'extra_numeric', 'types', 'queued_types',
                'cells', 'own', 'map_visibility', 'scalars', 'player_identity',
                'map_static', 'low', 'span', 'build_mask', 'build_signature', 'build_features',
                'task_numeric', 'task_relations', 'task_relation_valid', 'task_relation_features',
                'task_commands', 'task_command_templates', 'task_command_template_present',
                'entity_upgrades', 'player_flags', 'purchasable_sciences')
MASK_FIELDS = ('owners', 'visible', 'verb', 'actor', 'unit_masks', 'target',
               'template', 'command', 'science', 'slot', 'cell', 'offset')
LIST_FIELDS = ('unit_choices', 'selected')


class Column:
    """Ragged arrays, deduplicated within a bounded chunk; no dense padding."""
    def __init__(self, data, ids, offsets, shapes, orders):
        self.data, self.ids, self.offsets = data, ids, offsets
        self.shapes, self.orders = shapes, orders

    @classmethod
    def build(cls, values, *, float_dtype=None):
        if float_dtype is not None:
            if np.dtype(float_dtype) != np.dtype('float32'):
                raise ValueError('Training storage precision must be float32 or unchanged')
            # Narrow before deduplication/concatenation, so fresh shards never
            # allocate a second float64 column buffer just to serialize it.
            with np.errstate(over='raise'):
                values = [None if value is None else
                          np.asarray(value).astype(np.float32, copy=False)
                          if np.asarray(value).dtype == np.dtype('float64') else np.asarray(value)
                          for value in values]
        present = [np.asarray(value) for value in values if value is not None]
        if not present:
            return None
        dtype, ndim = present[0].dtype, present[0].ndim
        _dtype(dtype)
        ids, offsets, shapes, orders, buffers, lookup = [], [0], [], [], [], {}
        for value in values:
            if value is None:
                ids.append(-1); continue
            value = np.asarray(value)
            if value.dtype != dtype or value.ndim != ndim:
                raise ValueError('Training column changes dtype or rank')
            order = 'F' if value.flags.f_contiguous and not value.flags.c_contiguous else 'C'
            raw = value.tobytes(order=order)
            key = (value.shape, order, hashlib.sha256(raw).digest())
            if key not in lookup:
                lookup[key] = len(buffers)
                buffers.append(np.frombuffer(raw, dtype=dtype))
                offsets.append(offsets[-1]+value.size)
                shapes.append(value.shape); orders.append(order == 'F')
            ids.append(lookup[key])
        def immutable(array):
            return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        return cls(*(immutable(array) for array in (np.concatenate(buffers), np.asarray(ids, dtype=np.int32),
            np.asarray(offsets, dtype=np.int64), np.asarray(shapes, dtype=np.int64).reshape(len(buffers), ndim),
            np.asarray(orders, dtype=bool))))

    def at(self, row):
        index = int(self.ids[row])
        if index < 0:
            return None
        return self.data[self.offsets[index]:self.offsets[index+1]].reshape(
            tuple(self.shapes[index]), order='F' if self.orders[index] else 'C')

    def select(self, start, stop):
        return Column.build([self.at(i) for i in range(start, stop)])


def encode_columns(columns, metadata, *, float_dtype=None):
    """One compressed member per chunk: small schema header + typed buffers."""
    if float_dtype is not None and np.dtype(float_dtype) != np.dtype('float32'):
        raise ValueError('Training storage precision must be float32 or unchanged')
    blocks, position, descriptions = [], 0, {}
    for name, column in columns.items():
        parts = {}
        for field in ('data', 'ids', 'offsets', 'shapes', 'orders'):
            array = np.asarray(getattr(column, field))
            if field == 'data' and float_dtype is not None and array.dtype == np.dtype('float64'):
                # Narrow feature values only. IDs, ragged offsets, shapes and
                # legality masks retain their exact integer/boolean types.
                with np.errstate(over='raise'):
                    array = array.astype(np.float32)
            packed = array.dtype == np.bool_
            raw = np.packbits(array.reshape(-1), bitorder='little').tobytes() if packed else array.tobytes()
            parts[field] = [array.dtype.str, list(array.shape), position, len(raw), packed]
            blocks.append(raw); position += len(raw)
        descriptions[name] = parts
    document = {'version': VERSION, 'columns': descriptions, 'rows': metadata}
    if float_dtype is not None:
        document['float_dtype'] = 'float32'
    header = _json_bytes(document)
    return struct.pack('<Q', len(header))+header+b''.join(blocks)


def decode_columns(raw):
    if len(raw) < 8:
        raise ValueError('Truncated training column header')
    length, = struct.unpack_from('<Q', raw)
    if length > len(raw)-8:
        raise ValueError('Invalid training column header length')
    header = json.loads(raw[8:8+length])
    if header['version'] != VERSION:
        raise ValueError('Unsupported training column version')
    base, columns, row_count = 8+length, {}, len(header['rows'])
    for name, parts in header['columns'].items():
        values = {}
        for field in ('data', 'ids', 'offsets', 'shapes', 'orders'):
            dtype, shape, offset, size, packed = parts[field]
            dtype = _dtype(dtype)
            if (not all(type(n) is int and n >= 0 for n in shape)
                    or offset < 0 or size < 0 or base+offset+size > len(raw)):
                raise ValueError('Training array outside buffer')
            count = int(np.prod(shape, dtype=object))
            if packed:
                if dtype != np.bool_ or size != (count+7)//8:
                    raise ValueError('Invalid packed training mask')
                bits = np.unpackbits(np.frombuffer(raw, np.uint8, count=size, offset=base+offset),
                                     count=count, bitorder='little')
                array = np.frombuffer(bits.tobytes(), dtype=bool)
            else:
                if size != count*dtype.itemsize:
                    raise ValueError('Training array size mismatch')
                array = np.frombuffer(raw, dtype=dtype, count=count, offset=base+offset)
            values[field] = array.reshape(shape)
        column = Column(**values)
        if (column.ids.shape != (row_count,) or column.ids.dtype.kind != 'i'
                or column.offsets.ndim != 1 or column.offsets.dtype.kind != 'i'
                or column.shapes.ndim != 2 or column.shapes.dtype.kind != 'i'
                or column.orders.dtype != np.bool_ or column.orders.shape != (len(column.shapes),)
                or len(column.offsets) != len(column.shapes)+1 or column.offsets[0] != 0
                or column.offsets[-1] != len(column.data)
                or (np.diff(column.offsets) < 0).any()
                or ((column.ids < -1) | (column.ids >= len(column.shapes))).any()
                or (column.shapes < 0).any()
                or not np.array_equal(np.diff(column.offsets), np.prod(column.shapes, axis=1, dtype=np.int64))):
            raise ValueError('Invalid ragged training column')
        columns[name] = column
    return columns, header['rows']


class TensorRows(Sequence):
    """Small row adapters over column buffers; no per-entity dictionaries."""
    def __init__(self, columns, metadata, common, key):
        self.columns, self.metadata, self.common, self.key = columns, metadata, common, key
        self.views = {}
        self.variants = {}
        for name, column in columns.items():
            if '|' in name:
                self.variants.setdefault(name.partition('|')[0], []).append(column)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            columns = {name: value for name, column in self.columns.items()
                       if (value := column.select(start, stop)) is not None}
            return TensorRows(columns, self.metadata[start:stop], self.common, self.key)
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError('Training row outside chunk')
        if item not in self.views:
            meta = self.metadata[item]
            encoded = {name: value for name in ARRAY_FIELDS if 'e/'+name in self.columns
                       and (value := self.columns['e/'+name].at(item)) is not None}
            variants = {}
            for name, columns in self.variants.items():
                values = [value for column in columns if (value := column.at(item)) is not None]
                if len(values) > 1:
                    raise ValueError('Overlapping training column variants')
                if values:
                    variants[name] = values[0]
            encoded.update({name[2:]: value for name, value in variants.items() if name.startswith('e/')})
            encoded.update(meta['encoded'])
            for name in COMMON_FIELDS:
                if name not in encoded and name in self.common:
                    encoded[name] = self.common[name]
            target = meta['row']['target']
            mask = {'key': self.key, 'teacher': target}
            mask.update({name[2:]: value for name, value in variants.items() if name.startswith('l/')})
            for name in MASK_FIELDS:
                if 'l/'+name in self.columns:
                    value = self.columns['l/'+name].at(item)
                    if value is not None:
                        mask[name] = value
            for name in LIST_FIELDS:
                if 'l/'+name in self.columns:
                    value = self.columns['l/'+name].at(item)
                    if value is not None:
                        mask[name] = value.tolist()
            encoded['_training_legality'] = mask
            self.views[item] = {**meta['row'], 'encoded': encoded}
        return self.views[item]

    def __reduce__(self):
        # Drop lazily created row adapters; transport typed columns only.
        return TensorRows, (self.columns, self.metadata, self.common, self.key)


def project(rows, common, policy, key, *, float_dtype=None):
    values, metadata = {}, []
    for row in rows:
        encoded, legality = row['encoded'], row['encoded']['_training_legality']
        if legality['key'] != key or legality['teacher'] != row['target']:
            raise ValueError('Column projection legality differs from teacher')
        meta = {'row': {name: row[name] for name in ('target', 'hold_ticks', 'scope') if name in row},
                'encoded': {'entity_count': len(encoded['entities']), 'rotated': bool(encoded['rotated'])}}
        for name in ARRAY_FIELDS:
            value = encoded.get(name)
            if name in COMMON_FIELDS and value is not None and name in common:
                if value.dtype == common[name].dtype and value.shape == common[name].shape and value.tobytes() == common[name].tobytes():
                    continue
            if value is not None:
                values.setdefault('e/'+name, [None]*len(rows))[len(metadata)] = value
        for name in MASK_FIELDS+LIST_FIELDS:
            if name in legality:
                value = legality[name]
                if name in LIST_FIELDS:
                    value = np.asarray(value, dtype=np.int64)
                values.setdefault('l/'+name, [None]*len(rows))[len(metadata)] = value
        metadata.append(meta)
    columns = {}
    for name, items in values.items():
        signatures = list(dict.fromkeys((np.asarray(item).dtype.str, np.asarray(item).ndim)
                                       for item in items if item is not None))
        if len(signatures) == 1:
            columns[name] = Column.build(items, float_dtype=float_dtype)
        else:
            # Empty-entity dummy features can be integer zero arrays while real
            # entities use floats. Retain every dtype/rank rather than casting.
            for number, signature in enumerate(signatures):
                selected = [item if item is not None and
                    (np.asarray(item).dtype.str, np.asarray(item).ndim) == signature else None for item in items]
                columns[name+'|'+str(number)] = Column.build(selected, float_dtype=float_dtype)
    return TensorRows(columns, metadata, common, key)
