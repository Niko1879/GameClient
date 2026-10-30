"""Per-player replay views in canonical 60 Hz time, without retaining complete games."""
from __future__ import annotations

from replay_actions import iter_demonstrations


def normalize_demonstration(row, rate):
    if rate == 60:
        return row
    result = {**row, 'original_frame': row['frame'], 'frame': row['frame']*60//rate,
              'hold_ticks': None if row['hold_ticks'] is None else row['hold_ticks']*60//rate}
    if 'observation' in row:  # light quiet samples carry none
        observation = row['observation']
        result['observation'] = {**observation, **{key: [{**entity,
            'last_seen_frame': entity['last_seen_frame']*60//rate} for entity in observation[key]]
            for key in ('own_entities', 'known_nonowned_entities')}}
    return result


class ReplayRows:
    """Reiterable player view; command adaptation stays in original source time."""
    def __init__(self, path, player, length, rate, *, validated=False):
        self.path, self.player, self.length, self.rate = path, player, length, rate
        self.validated = validated

    def __len__(self):
        return self.length

    def __iter__(self):
        for row in iter_demonstrations(self.path, validate=False, player=self.player, validated=self.validated):
            if row['player_index'] == self.player:
                yield normalize_demonstration(row, self.rate)
