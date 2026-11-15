"""The opponent's side in a recorded China match, as both players saw it in the lobby.

Shards do not store it. The replay header has both templates, except for random
picks, which the match listing resolves (retail 1.04 replays: the game seed,
random_sides.py). Since 13 Nov 2026 the exported players are vanilla, Tank or Nuke China, and both
players of a mixed China match are exported, so a match stores both factions (match_factions) and
each row's enemy is the faction that is not its own (event_dataset.read).
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


def _natives(sides):
    """Both native faction names when every side is resolved (no Random), else None."""
    natives = [NATIVE.get(side) for side in sides]
    return natives if len(natives) == 2 and all(natives) else None


def match_factions(replay):
    """[native faction, native faction] of the two players (header order), or None when unresolved."""
    replay = Path(replay)
    with replay.open('rb') as stream:
        header = parse_replay(stream.read(HEADER_BYTES))
    found = _natives([player['header_faction'] for player in header['players']])
    if found:
        return found
    sidecar = replay.with_suffix('.json')
    if sidecar.is_file():
        found = _natives(json.loads(sidecar.read_text(encoding='utf-8')).get('listing_factions') or ())
        if found:
            return found
    # Retail archive replays have no listing; their game seed resolves Random.
    from random_sides import RETAIL_RELEASES, resolved_factions
    if header['exe_crc'] in RETAIL_RELEASES:
        return _natives(resolved_factions(header))
    return None


def enemy_of(own, factions):
    """The enemy's native faction for a player of faction `own` in a match of `factions`, or None."""
    if not factions or own not in factions:
        return None
    rest = list(factions)
    rest.remove(own)
    return rest[0]
