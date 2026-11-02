"""The opponent's side in a recorded China match, as both players saw it in the lobby.

Shards do not store it. The replay header has both templates, except for random
picks, which the match listing resolves. The controlled player is vanilla China,
so the enemy is whatever remains after removing one vanilla China.
"""
from __future__ import annotations

import json
from pathlib import Path

from harvest_china_replays import parse_replay

NATIVE = {'USA': 'FactionAmerica', 'China': 'FactionChina', 'GLA': 'FactionGLA',
          'USA Super Weapon': 'FactionAmericaSuperWeaponGeneral', 'USA Laser': 'FactionAmericaLaserGeneral',
          'USA Airforce': 'FactionAmericaAirForceGeneral', 'China Tank': 'FactionChinaTankGeneral',
          'China Infantry': 'FactionChinaInfantryGeneral', 'China Nuke': 'FactionChinaNukeGeneral',
          'GLA Toxin': 'FactionGLAToxinGeneral', 'GLA Demo': 'FactionGLADemolitionGeneral',
          'GLA Stealth': 'FactionGLAStealthGeneral'}
HEADER_BYTES = 65536


def _other(sides):
    sides = list(sides)
    if 'China' not in sides:
        return None
    sides.remove('China')
    return NATIVE.get(sides[0])


def enemy_faction(replay):
    """The enemy's native faction name, or None when the records leave it unresolved."""
    replay = Path(replay)
    with replay.open('rb') as stream:
        header = parse_replay(stream.read(HEADER_BYTES))
    enemy = _other(player['header_faction'] for player in header['players'])
    if enemy:
        return enemy
    sidecar = replay.with_suffix('.json')
    if not sidecar.is_file():
        return None
    return _other(json.loads(sidecar.read_text(encoding='utf-8')).get('listing_factions') or ())
