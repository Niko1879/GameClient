"""Random sides in a retail Zero Hour 1.04 header, resolved from the game seed.

The header keeps template -1 for a Random pick; the game resolves it at start in
GameLogic.cpp populateRandomSideAndColor with the logic RNG seeded by SD=. Matched
the engine on 9 of 9 gamereplays.org replays (6 Nov 2026). Only retail releases use
it: Generals Online matches resolve Random from their listing instead.
"""
from __future__ import annotations

import re

# Retail 1.04 executables in the gamereplays.org archive (INI FEAAE3F3).
RETAIL_RELEASES = frozenset({'DA2B4B18', 'C5BD7FC3'})
_MASK = 0xFFFFFFFF
_INIT = (0xf22d0e56, 0x883126e9, 0xc624dd2f, 0x0702c49c, 0x9e353f7d, 0x6fdf3b64)
_START_TEMPLATES = list(range(2, 14))  # playable, unlocked (Boss general excluded)
_COLORS = 8
_HUMAN = re.compile(r"H(.*?),([0-9A-Fa-f]+),(\d+),([TF]{2}),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+):", re.S)
_AI = re.compile(r"C(\w),(-?\d+),(-?\d+),(-?\d+),(-?\d+):")


class _Random:
    """GameLogic's random value generator (Common/RandomValue.cpp)."""
    def __init__(self, seed):
        self.state = [(seed + v) & _MASK for v in _INIT]

    def value(self):
        s = self.state

        def add(a, b, carry):
            total = (a + b + carry) & _MASK
            return total, int(total < a or total < b)
        ax, carry = add(s[5], s[4], 0); s[4] = ax
        ax, carry = add(ax, s[3], carry); s[3] = ax
        ax, carry = add(ax, s[2], carry); s[2] = ax
        ax, carry = add(ax, s[1], carry); s[1] = ax
        ax, carry = add(ax, s[0], carry); s[0] = ax
        for i in (5, 4, 3, 2, 1):
            s[i] = (s[i] + 1) & _MASK
            if s[i]:
                break
        else:
            s[0] = (s[0] + 1) & _MASK
            ax = (ax + 1) & _MASK
        return ax

    def between(self, low, high):
        span = (high - low + 1) & _MASK
        return high if span == 0 else self.value() % span + low


def _slots(options):
    body = re.search(r"(?:^|;)S=(.*);$", options, re.S).group(1)
    slots, cursor = [], 0
    while cursor < len(body) and len(slots) < 8:
        if body[cursor:cursor + 2] in ('X:', 'O:'):
            slots.append(None)
            cursor += 2
            continue
        match = (_AI if body[cursor] == 'C' else _HUMAN).match(body, cursor)
        if not match:
            raise ValueError('Malformed slot list')
        groups = match.groups()
        color, template = (int(groups[1]), int(groups[2])) if body[cursor] == 'C' else (int(groups[4]), int(groups[5]))
        slots.append({'color': color, 'template': template})
        cursor = match.end()
    return slots


def resolved_templates(header):
    """{slot: resolved template ID} for every occupied slot (observers stay -2)."""
    if header['exe_crc'] not in RETAIL_RELEASES:
        raise ValueError('Seed resolution is verified on retail 1.04 only')
    options = header['game_options']
    seed = int(re.search(r"(?:^|;)SD=(\d+);", options).group(1))
    rng, slots = _Random(seed), _slots(options)
    taken = {s['color'] for s in slots if s and s['color'] >= 0}
    resolved = {}
    for index, slot in enumerate(slots):
        if not slot:
            continue
        template = slot['template']
        while template != -2 and (template < 0 or template >= 15):
            for _ in range(seed % 7):
                rng.between(0, 1)
            template = _START_TEMPLATES[rng.between(0, 1000) % len(_START_TEMPLATES)]
        resolved[index] = template
        if slot['color'] < 0:  # random colours draw from the same generator
            color = -1
            while color == -1:
                color = rng.between(0, _COLORS - 1)
                if color in taken:
                    color = -1
            slot['color'] = color
            taken.add(color)
    return resolved


def resolved_factions(header):
    """Listing-style faction name per header player, in header order."""
    from harvest_china_replays import TEMPLATES
    resolved = resolved_templates(header)
    return [TEMPLATES.get(resolved[player['slot']], 'Unknown') for player in header['players']]
