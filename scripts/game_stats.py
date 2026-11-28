"""Per-player game statistics accumulated step by step, so a head-to-head needs no trajectory file.

play_head_to_head reports them per side: dozers, Command Center, income, production, and every order kind.
"""
from __future__ import annotations

from collections import Counter

FPS = 60  # Stored frames are canonical 60 Hz.


class SideStats:
    """One player's running totals over transitions (observation before and after a step)."""

    def __init__(self, player):
        self.player = str(player)
        self.dozers, self.max_dozers, self.last_dozer, self.cc_gone = set(), 0, None, None
        self.income, self.units, self.seen, self.orders = 0, 0, set(), Counter()

    def update(self, observations, next_observations, next_frame, feedback, actions=None):
        player = self.player
        before, after = observations[player], next_observations[player]
        actions = actions or {}
        # Live steps key actions by the engine's player number; a trajectory read back from JSON, by its string.
        for action in actions.get(int(player)) or actions.get(player) or ():
            if isinstance(action, dict):
                self.orders[action.get('kind')] += 1
        sold = any(f['accepted'] and f['action']['kind'] == 'sell' for f in feedback if str(f['player']) == player)
        if not sold:
            self.income += max(0, after['player']['cash']-before['player']['cash'])
        entities = after['own_entities']
        alive = [e['id'] for e in entities if e['type'].endswith('VehicleDozer')]
        self.dozers.update(alive)
        self.max_dozers = max(self.max_dozers, len(alive))
        minute = next_frame/FPS/60
        if alive:
            self.last_dozer = None
        elif self.last_dozer is None:
            self.last_dozer = minute
        if self.cc_gone is None and not any(e['type'].endswith('CommandCenter') for e in entities):
            self.cc_gone = minute
        for e in entities:
            if e['id'] not in self.seen:
                self.seen.add(e['id'])
                if e.get('producer_id') and not e['type'].endswith(('Mine', 'ListeningOutpost')):
                    self.units += 1

    def result(self):
        return dict(dozers_ever=len(self.dozers), max_dozers=self.max_dozers, no_dozer_from_min=self.last_dozer,
                    no_cc_from_min=self.cc_gone, income_proxy=self.income, units_produced=self.units,
                    orders=dict(self.orders))
