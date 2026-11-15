"""Player-slot counts of maps, for keeping only 2-slot (1v1) maps in training.

A map's slot count is its number of start positions: waypoints named
Player_1_Start, Player_2_Start, ... counted consecutively from 1, as the engine's
MapCache does (numPlayers). Two sources, keyed by the map CRC replays record:

- MapCache.ini files the engine wrote (official maps live inside .big archives,
  so this is the only source for them; official entries are also kept by name,
  because patched official maps can carry another CRC);
- .map files on disk (EAR/RefPack-compressed or raw), read directly. The
  retail map store is laid out as <CRC>/<name>/<name>.map.

Checked 10 Nov 2026: direct counts equal MapCache's numPlayers on all 51 maps
both cover. "[RANK]" in a name is GenTool's ranked-pack label and includes team
maps (e.g. "[RANK] [NMC 2x2] ..."), so it says nothing about slots.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT/'replay-runtime/map-slots.json'
MAP_CACHES = [ROOT/'replay-runtime/game/Maps/MapCache.ini', ROOT/'replay-runtime/retail-104/game/Maps/MapCache.ini',
              ROOT/'replay-runtime/profile/Maps/MapCache.ini']
RETAIL_MAPS = ROOT/'replay-runtime/retail-104/maps'


def refpack(data):
    """EA RefPack decompression (the payload after an 'EAR\\0' + size header)."""
    flags, pos = data[0], 2
    size_bytes = 4 if flags & 0x80 else 3
    if flags & 0x01:
        pos += size_bytes  # compressed size, unused
    expected = int.from_bytes(data[pos:pos+size_bytes], 'big')
    pos += size_bytes
    out = bytearray()
    while pos < len(data):
        b0 = data[pos]
        if b0 < 0x80:
            b1 = data[pos+1]
            pos += 2
            literal, distance, length = b0 & 3, ((b0 & 0x60) << 3)+b1+1, ((b0 & 0x1C) >> 2)+3
        elif b0 < 0xC0:
            b1, b2 = data[pos+1], data[pos+2]
            pos += 3
            literal, distance, length = b1 >> 6, ((b1 & 0x3F) << 8)+b2+1, (b0 & 0x3F)+4
        elif b0 < 0xE0:
            b1, b2, b3 = data[pos+1], data[pos+2], data[pos+3]
            pos += 4
            literal, distance, length = b0 & 3, ((b0 & 0x10) << 12)+(b1 << 8)+b2+1, ((b0 & 0x0C) << 6)+b3+5
        else:
            pos += 1
            literal = ((b0 & 0x1F) << 2)+4 if b0 < 0xFC else b0 & 3
            out += data[pos:pos+literal]
            pos += literal
            if b0 >= 0xFC:
                break
            continue
        out += data[pos:pos+literal]
        pos += literal
        start = len(out)-distance
        for i in range(length):  # Overlapping copies repeat bytes, so copy one at a time.
            out.append(out[start+i])
    if len(out) != expected:
        raise ValueError(f'RefPack output is {len(out)} bytes, header says {expected}')
    return bytes(out)


def map_bytes(path):
    raw = Path(path).read_bytes()
    return refpack(raw[8:]) if raw[:4] == b'EAR\0' else raw


def start_count(path):
    """Consecutive Player_N_Start waypoints from 1 (MapCache's numPlayers)."""
    present = {int(n) for n in re.findall(rb'Player_(\d)_Start', map_bytes(path))}
    count = 0
    while count+1 in present:
        count += 1
    return count


def _unescape(key):
    return re.sub(r'_([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), key)


def map_cache(path):
    """[(map path, '%08X' CRC, numPlayers, official)] of one MapCache.ini."""
    entries, current = [], None
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        line = line.strip()
        if line.startswith('MapCache '):
            current = {'name': _unescape(line[len('MapCache '):]).replace('\\', '/').casefold()}
        elif current is not None and '=' in line:
            key, _, value = (part.strip() for part in line.partition('='))
            current[key] = value
        elif line == 'END' and current is not None:
            entries.append((current['name'], '%08X' % int(current['fileCRC']), int(current['numPlayers']),
                            current.get('isOfficial') == 'yes'))
            current = None
    return entries


def official_key(map_path):
    """'maps/alpine assault' -> 'alpine assault' (a replay header's official map path)."""
    return map_path.replace('\\', '/').rstrip('/').split('/')[-1].casefold()


def build(caches=MAP_CACHES, retail_maps=RETAIL_MAPS):
    by_crc, official, sources = {}, {}, {'map_cache': 0, 'map_file': 0}
    for cache in caches:
        if not Path(cache).is_file():
            continue
        for name, crc, slots, is_official in map_cache(cache):
            by_crc[crc] = slots
            sources['map_cache'] += 1
            if is_official and name.startswith('maps/') and name.endswith('.map'):
                official[name[:-len('.map')].split('/')[-1]] = slots
    for path in sorted(Path(retail_maps).glob('*/*/*.map')):
        crc = path.parts[-3].upper()
        slots = start_count(path)
        if by_crc.get(crc, slots) != slots:
            raise ValueError(f'{path}: {slots} start waypoints, MapCache says {by_crc[crc]}')
        by_crc[crc] = slots
        sources['map_file'] += 1
    return {'by_crc': dict(sorted(by_crc.items())), 'official': dict(sorted(official.items())), 'sources': sources}


def load(path=TABLE):
    if not Path(path).is_file():
        raise FileNotFoundError(f'{path} is missing; run replay_cleaning.py map-slots')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def slots(table, map_path, crc):
    """The map's slot count, or None when no source covers it."""
    found = table['by_crc'].get(crc)
    if found is None and map_path.casefold().startswith('maps/'):
        found = table['official'].get(official_key(map_path))
    return found
