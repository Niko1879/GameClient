"""GOBS1 transport: shared field layouts and typed scalar columns, read lazily.

This changes storage only. The policy still receives ordinary Python values at
the point an event is encoded. Headers/commands and dictionary deltas are small
JSON envelopes; dense observations never pass through a JSON text encoder.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import json
import math
import os
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

MAGIC = b'GOBS1\r\n\x1a'
BLOCK = struct.Struct('<4s6I')
MAX_BLOCK = 512*1024**2


def materialize(value):
    if isinstance(value, DeferredObservation):
        return materialize(value.loaded())
    if isinstance(value, ObjectView):
        return value.frame.build()[value.node]
    if isinstance(value, EntityList):
        return value.plain()
    return value


class DeferredObservation(Mapping):
    """Used only after a complete source-bound audit, never by that audit."""
    __slots__ = ('loader', 'view')

    def __init__(self, loader):
        self.loader, self.view = loader, None

    def loaded(self):
        if self.view is None:
            self.view = self.loader()
            self.loader = None
        return self.view

    def __getitem__(self, key):
        return self.loaded()[key]

    def __iter__(self):
        return iter(self.loaded())

    def __len__(self):
        return len(self.loaded())


class ObjectView(Mapping):
    __slots__ = ('frame', 'node', 'start', 'fields', 'cache', 'observation')

    def __init__(self, frame, node, *, observation=False):
        bits = frame.values[node]
        self.frame, self.node, self.start = frame, node, bits & 0xffffffff
        self.fields = frame.schemas[bits >> 32]
        self.cache, self.observation = {}, observation

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    # Membership comes from the field layout; it never decodes a value.
    def __contains__(self, key):
        return key in self.fields

    def get(self, key, default=None):
        return self[key] if key in self.fields else default

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        node = self.start+self.fields[key]
        if self.observation and key in ('own_entities', 'known_nonowned_entities'):
            if self.frame.tags[node] != 7:
                raise ValueError('Binary entity table is not an array')
            result = EntityList(self.frame, node)
        else:
            result = self.frame.value(node)
        self.cache[key] = result
        return result


class EntityList:
    __slots__ = ('frame', 'start', 'count')

    def __init__(self, frame, node):
        bits = frame.values[node]
        self.frame, self.start, self.count = frame, bits & 0xffffffff, bits >> 32

    def __len__(self):
        return self.count

    def __iter__(self):
        for node in range(self.start, self.start+self.count):
            if self.frame.tags[node] != 8:
                raise ValueError('Binary entity is not an object')
            yield ObjectView(self.frame, node)

    def __add__(self, other):
        return list(self)+list(other)

    def plain(self):
        """Ordinary entity dicts, equal to Frame.value per entity, with scalar fields inlined."""
        frame = self.frame
        tags, values, floats, strings = frame.tags, frame.values, frame.floats, frame.strings
        schemas, value = frame.schemas, frame.value
        result = []
        for node in range(self.start, self.start+self.count):
            if tags[node] != 8:
                raise ValueError('Binary entity is not an object')
            bits = values[node]
            base, entity = bits & 0xffffffff, {}
            for key, offset in schemas[bits >> 32].items():
                child = base+offset
                tag = tags[child]
                if tag == 5:
                    entity[key] = floats[child]
                elif tag == 4:
                    entity[key] = values[child]
                elif tag == 6:
                    entity[key] = strings[values[child]]
                else:
                    entity[key] = value(child)
            result.append(entity)
        return result


    def field_nodes(self, name):
        nodes = np.arange(self.start, self.start+self.count)
        if np.any(self.frame.tag_array[nodes] != 8):
            raise ValueError('Binary entity is not an object')
        entries = self.frame.value_array[nodes]
        offsets = self.frame.schemas.offsets(name)[(entries >> np.uint64(32)).astype(np.int64)]
        if np.any(offsets < 0):
            raise KeyError(name)
        return (entries & np.uint64(0xffffffff)).astype(np.int64)+offsets

    def field_values(self, name):
        nodes = self.field_nodes(name)
        tags, values = self.frame.tag_array[nodes], self.frame.value_array[nodes]
        if np.all(tags == 6):
            return [self.frame.strings[i] for i in values.tolist()]
        if np.all(tags == 4):
            return values.tolist()
        return [self.frame.value(int(node)) for node in nodes]

    def ids(self, *, visible_only=False):
        if not self.count:
            return set()
        nodes = self.field_nodes('id')
        if visible_only:
            visible = self.field_nodes('visible')
            tags = self.frame.tag_array[visible]
            mask = (tags == 2) if np.all((tags == 1) | (tags == 2)) else np.asarray(
                [bool(self.frame.value(int(node))) for node in visible])
            nodes = nodes[mask]
        if np.all(self.frame.tag_array[nodes] == 4):
            return set(self.frame.value_array[nodes].tolist())
        return {self.frame.value(int(node)) for node in nodes}


class Layouts(list):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def offsets(self, field):
        cached = self.cache.get(field)
        if cached is None or len(cached) != len(self):
            cached = np.asarray([layout.get(field, -1) for layout in self], dtype=np.int64)
            self.cache[field] = cached
        return cached


class Frame:
    def __init__(self, raw, count, strings, schemas, *, check_finite=False, trusted=False):
        """`trusted` skips the structural checks for a block the audit checks or has checked."""
        self.raw, self.strings, self.schemas = raw, strings, schemas
        data = memoryview(raw)
        self.tags = data[:count]
        self.values = data[count:].cast('Q')
        self.floats = data[count:].cast('d')
        self.signed = data[count:].cast('q')
        tags = np.frombuffer(self.tags, np.uint8)
        values = np.frombuffer(data[count:], '<u8')
        self.tag_array, self.value_array = tags, values
        if trusted:
            return
        if not count or np.any(tags > 8) or self.tags[0] != 8:
            raise ValueError('Invalid binary observation scalar tag/root')
        if check_finite and not np.isfinite(np.frombuffer(data[count:], '<f8')[tags == 5]).all():
            raise ValueError('Nonfinite feature')
        if np.any(values[tags == 6] >= len(strings)):
            raise ValueError('Invalid binary string reference')
        objects, arrays = tags == 8, tags == 7
        layout_ids = values[objects] >> np.uint64(32)
        if np.any(layout_ids >= len(schemas)):
            raise ValueError('Invalid binary field-layout reference')
        # All children follow their parent. Sorted, contiguous child ranges
        # cover every non-root node exactly once, forbidding cycles/aliases.
        containers = objects | arrays
        nodes = np.flatnonzero(containers)
        starts = (values[containers] & np.uint64(0xffffffff)).astype(np.int64)
        counts = (values[containers] >> np.uint64(32)).astype(np.int64)
        counts[objects[containers]] = np.asarray([len(schemas[int(i)]) for i in layout_ids])
        active = counts > 0
        if np.any(starts[active] <= nodes[active]) or np.any(starts+counts > count):
            raise ValueError('Invalid binary child range')
        order = np.argsort(starts[active])
        first, lengths = starts[active][order], counts[active][order]
        if count > 1 and (not len(first) or first[0] != 1 or first[-1]+lengths[-1] != count
                         or np.any(first[1:] != first[:-1]+lengths[:-1])):
            raise ValueError('Binary observation contains overlapping/orphan nodes')

    def build(self):
        """Every node as ordinary Python values: value(node) for all nodes in one pass."""
        tags, bits = self.tag_array, self.value_array
        nodes = bits.astype(object)  # unsigned integers, and containers' child ranges
        floats = tags == 5
        nodes[floats] = np.frombuffer(self.raw, '<f8', offset=len(tags))[floats]
        signed = tags == 3
        nodes[signed] = bits[signed].astype('<i8')
        text = tags == 6
        strings = self.strings
        nodes[text] = [strings[i] for i in bits[text].tolist()]
        nodes[tags == 2] = True
        nodes[tags == 1] = False
        nodes[tags == 0] = None
        nodes = nodes.tolist()
        schemas = self.schemas
        # All children follow their parent, so a backward pass finds them built.
        containers = np.flatnonzero(tags >= 7)[::-1]
        for node, tag in zip(containers.tolist(), tags[containers].tolist()):
            packed = nodes[node]
            start, size = packed & 0xffffffff, packed >> 32
            if tag == 7:
                nodes[node] = nodes[start:start+size]
            else:
                fields = schemas[size]
                nodes[node] = dict(zip(fields, nodes[start:start+len(fields)]))
        return nodes

    def value(self, node):
        tag, bits = self.tags[node], self.values[node]
        if tag == 0:
            return None
        if tag <= 2:
            return tag == 2
        if tag == 3:
            return bits-(1 << 64) if bits & (1 << 63) else bits
        if tag == 4:
            return bits
        if tag == 5:
            return self.floats[node]
        if tag == 6:
            return self.strings[bits]
        start, size = bits & 0xffffffff, bits >> 32
        if tag == 7:
            if size >= 16:
                child_tag = self.tags[start]
                if child_tag in (3, 4, 5) and np.all(self.tag_array[start:start+size] == child_tag):
                    column = self.floats if child_tag == 5 else self.signed if child_tag == 3 else self.values
                    return column[start:start+size].tolist()
            return [self.value(child) for child in range(start, start+size)]
        return {key:self.value(start+offset) for key, offset in self.schemas[size].items()}


def load_block(stored, text, count, raw_size, crc, flags, strings, schemas, check_finite, trusted=False):
    if flags:
        inflater = zlib.decompressobj()
        try:
            raw = inflater.decompress(stored, raw_size+1)
        except zlib.error as error:
            raise ValueError('Invalid compressed binary observation') from error
        if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError('Invalid compressed binary observation')
    else:
        raw = stored
    if len(raw) != raw_size or (not trusted and zlib.crc32(raw, zlib.crc32(text)) != crc):
        raise ValueError('Binary block checksum/size mismatch')
    if count:
        frame = Frame(raw, count, strings, schemas, check_finite=check_finite, trusted=trusted)
        return ObjectView(frame, 0, observation=True)


def trim_export(source, destination, last_frame, crc_comparisons):
    """Turn a CRC-diverged export into a finished one ending at `last_frame`.

    Blocks are self-contained apart from the dictionaries earlier blocks grow, so
    the kept rows are a byte prefix. The file is cut in place, not rewritten.
    """
    source, destination = Path(source), Path(destination)
    samples = commands = waits = 0
    cut = footer = None
    with source.open('r+b') as stream:
        if stream.read(8) != MAGIC:
            raise ValueError('Unsupported binary replay signature/version')
        while footer is None:
            offset = stream.tell()
            header = stream.read(BLOCK.size)
            if len(header) != BLOCK.size:
                raise ValueError('Diverged export has no footer')
            _, meta_size, _, _, stored_size, _, _ = BLOCK.unpack(header)
            record = json.loads(stream.read(meta_size).decode('utf-8'))['record']
            stream.seek(stored_size, 1)
            if record['record'] == 'footer':
                footer = record
            elif record['record'] == 'sample':
                if record['frame'] > last_frame:
                    cut = offset if cut is None else cut
                elif cut is not None:
                    raise ValueError('Export samples are not in frame order')
                else:
                    samples += 1
                    commands += len(record['actions'])
                    waits += not record['actions']
        if footer['valid'] or not footer['crc_mismatch']:
            raise ValueError('Only a CRC-diverged export can be trimmed')
        # These counters describe the removed rows too; audit treats them as optional.
        kept = {k:v for k, v in footer.items() if k not in ('observed_sighting_removals', 'temporary_sightings_forgotten')}
        kept.update(valid=True, samples=samples, commands=commands, wait_samples=waits, final_frame=last_frame,
                    verified_through_frame=last_frame, recorded_crc_messages_compared=crc_comparisons,
                    result='recorded simulation CRC checks passed through verified_through_frame')
        metadata = json.dumps(dict(record=kept, strings=[], schemas=[])).encode('utf-8')
        stream.seek(offset if cut is None else cut)
        stream.truncate()
        stream.write(BLOCK.pack(b'BLK1', len(metadata), 0, 0, 0, zlib.crc32(b'', zlib.crc32(metadata)), 0))
        stream.write(metadata)
    os.replace(source, destination)


def records(path, *, decoder=None, check_finite=False, lazy=False, player=None, validated=False):
    path = Path(path)
    with path.open('rb') as stream:
        total = path.stat().st_size  # a finished export; callers verify its digest
        yield from stream_records(stream, decoder=decoder, check_finite=check_finite, lazy=lazy,
                                  player=player, validated=validated, size=lambda: total)


def stream_records(stream, *, decoder=None, check_finite=False, lazy=False, player=None, validated=False,
                   size=None):
    """Records from a binary stream; a short read means end of stream.

    `size` (the total stream length) is needed only to skip other players.
    """
    if sys.byteorder != 'little':
        raise ValueError('Binary replay reader requires little-endian host')
    decoder = decoder or json.JSONDecoder()
    scan = decoder.raw_decode
    strings, schemas = [], Layouts()
    # Only a complete audit, which reads every block itself, may leave blocks unchecked.
    trusted = validated and lazy and not check_finite
    if stream.read(8) != MAGIC:
        raise ValueError('Unsupported binary replay signature/version')
    while True:
        header = stream.read(BLOCK.size)
        if not header:
            break
        if len(header) != BLOCK.size:
            raise ValueError('Truncated binary block header')
        magic, meta_size, count, raw_size, stored_size, crc, flags = BLOCK.unpack(header)
        if (magic != b'BLK1' or flags not in (0, 1) or meta_size > 16*1024**2
                or raw_size != count*9 or raw_size > MAX_BLOCK
                or stored_size > MAX_BLOCK or (flags == 0 and stored_size != raw_size)):
            raise ValueError('Invalid binary block dimensions/flags')
        text = stream.read(meta_size)
        if len(text) != meta_size:
            raise ValueError('Truncated binary metadata')
        decoded = text.decode('utf-8')
        envelope, end = scan(decoded)
        if end != len(decoded):
            raise ValueError('Invalid binary metadata envelope')
        if not isinstance(envelope, dict) or len(envelope) != 3 or not {'record', 'strings', 'schemas'} <= envelope.keys():
            raise ValueError('Invalid binary metadata envelope')
        added = envelope['strings']
        if not isinstance(added, list):
            raise ValueError('Invalid binary string dictionary')
        if added:
            if any(not isinstance(s, str) for s in added):
                raise ValueError('Invalid binary string dictionary')
            strings.extend(added)
        for layout in envelope['schemas']:
            if not isinstance(layout, list) or any(type(i) is not int or not 0 <= i < len(strings) for i in layout):
                raise ValueError('Invalid binary field layout')
            names = [strings[i] for i in layout]
            if len(set(names)) != len(names):
                raise ValueError('Duplicate binary object field')
            schemas.append({name:i for i, name in enumerate(names)})
        record = envelope['record']
        # Scheduled quiet capture writes some quiet samples as metadata only.
        observed = isinstance(record, dict) and record.get('record') == 'sample' and record.get('light') is not True
        if not isinstance(record, dict) or observed != bool(count) or record.get('light', True) is not True:
            raise ValueError('Binary observation/record mismatch')
        if player is not None and record.get('record') == 'sample' and record['player_index'] != player:
            # Complete audits verify every block. Player views use the
            # already source-hashed receipt and skip unrelated payloads.
            if stream.tell()+stored_size > size():
                raise ValueError('Truncated binary observation')
            stream.seek(stored_size, 1)
            continue
        stored = stream.read(stored_size)
        if len(stored) != stored_size:
            raise ValueError('Truncated binary observation')
        if count and trusted:
            record['observation'] = DeferredObservation(
                partial(load_block, stored, text, count, raw_size, crc, flags, strings, schemas, False, True))
        else:
            view = load_block(stored, text, count, raw_size, crc, flags, strings, schemas, check_finite)
            if count:
                record['observation'] = view if lazy else materialize(view)
        yield record
