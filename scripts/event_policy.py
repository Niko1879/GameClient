"""Event policy configuration and the deployed actor, which looks at a fixed interval."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from event_schedule import canonical_frame, native_deadline
from generals_identity import vanilla_china_production
from generals_policy import (ARCHITECTURE, OBSERVATION_VERSION, PLAY_APM, PLAY_RATING, UNRATED, TemplateTable, apm_inputs, initial_home,
                             opener_inputs, rating_inputs)
from nn_numpy import no_grad
from replay_sequences import normalize_demonstration

PLACEMENT_MODES = ('fast', 'full_mask')


def command_vocabulary(command_catalog):
    """Canonical commands, general's promotions and upgrades from the native catalogue."""
    commands = command_catalog['commands']
    return {'commands': sorted(name for name, info in commands.items() if info['kind'] in ('upgrade', 'ability')),
            'command_info': {name: [info['kind'], info['target']] for name, info in sorted(commands.items())},
            'command_menus': command_catalog['menus'], 'command_shortcuts': command_catalog['shortcuts'],
            'promotions': command_catalog['promotions'], 'upgrades': sorted(command_catalog['upgrades']),
            # Special powers, for v11 launch-sound cues (generals_policy.Encoder._effects).
            'powers': sorted({info['special_power'] for info in commands.values() if info.get('special_power')})}


def default_config(vocabulary, placement_mode='fast'):
    config = {'types': vocabulary['types'], 'templates': vocabulary['templates'], 'menus': vocabulary['menus'],
              'commands': vocabulary['commands'], 'command_info': vocabulary['command_info'],
              'command_menus': vocabulary['command_menus'], 'command_shortcuts': vocabulary['command_shortcuts'],
              'promotions': vocabulary['promotions'], 'upgrades': vocabulary['upgrades'], 'powers': vocabulary['powers'],
              'sciences': sorted({name for names in vocabulary['promotions'].values() for name in names}),
              'architecture': ARCHITECTURE, 'observation_policy_version': OBSERVATION_VERSION,
              # Widths are model-only and never change shard columns; the rest are generals_policy.WIDTHS.
              'embedding': 64, 'map_side': 16, 'map_hidden': 128, 'max_group': 64,
              'position_bins': 256, 'angle_bins': 16,
              # The exporter's quiet grid, and how many no-op snapshots a second the shards store.
              'event_observation_ms': 100, 'snapshot_rate': .5,
              'variant_l2': 1e-4,
              'argument_loss_weights': {'verb': 2.0, 'unit': .5, 'target': .5,
                                        'template': 1.0, 'command': 1.0, 'slot': 1.0,
                                        'cell': .75, 'offset': .5, 'angle': .1}}
    return placement_config(vanilla_china_production(config), placement_mode)


def frozen_config(catalog, placement_mode='fast'):
    """Fixed vocabulary from the engine handshake: every object type, menu and production choice."""
    menus = {key: sorted(values) for key, values in sorted(catalog['action_menus'].items())}
    return default_config({'types': ['<unknown>'] + sorted(set(catalog['template_catalog'].values()) - {'<unknown>'}),
                           'templates': sorted({name for values in menus.values() for name in values}),
                           'menus': menus, **command_vocabulary(catalog['command_catalog'])}, placement_mode)


def placement_config(config, mode):
    """Fast mode trains on accepted sites and snaps locally; full-mask mode uses exact native grids."""
    if mode not in PLACEMENT_MODES:
        raise ValueError('Placement mode must be fast or full_mask')
    return dict(config, placement_mode=mode, build_snap_radius=32.0 if mode == 'fast' else 0.0, build_snap_step=4.0)


SCIENCE_ORDERS = Path(__file__).resolve().parents[1]/'replay-runtime/science-orders.json'


class ScienceBuyer:
    """Scripted general's-promotion purchases: the model has no science action (re-export fix 2).

    One fixed priority list per opponent faction (replay-runtime/science-orders.json, made by
    scripts/science_presets.py from top-10% players; user sign-off 12 Nov 2026). The buyer takes the
    first science of the list it does not own and buys it once the engine lists it as purchasable.
    It never skips ahead: against GLA and China the list opens with a rank-3 science, so waiting is
    what saves the rank-1 point, as top players do. An unknown opponent gets the pooled default list.
    """
    def __init__(self, path=SCIENCE_ORDERS):
        document = json.loads(Path(path).read_text(encoding='utf-8'))
        if document.get('kind') != 'generals_science_orders':
            raise ValueError('Not a science orders file')
        self.orders, self.default = document['orders'], document['default']
        self.order = self.default

    def configure(self, faction):
        """The opponent's native faction name (EventActor.configure_enemy); None or unknown uses the default."""
        self.order = self.orders.get(faction, self.default)

    def __call__(self, observation):
        player = observation['player']
        owned = set(player.get('sciences', ()))
        wanted = next((name for name in self.order if name not in owned), None)
        if wanted is not None and wanted in player.get('purchasable_sciences', ()):
            return [{'kind': 'science', 'science': wanted}]
        return []


def science_buyer(faction):
    """The science buyer for a China general's own faction: science-orders.json holds vanilla China's
    trees, so Tank/Nuke sides buy no sciences (user, 15 Nov 2026)."""
    if faction == 'FactionChina':
        return ScienceBuyer()
    buyer = ScienceBuyer()
    buyer.orders, buyer.default = {}, []
    buyer.order = []
    return buyer


# Live play's default look interval (canonical ticks): 200 ms. Nothing about it is learned.
LOOK_TICKS = 12
# At most ten orders a second, whatever the look interval.
MIN_ORDER_TICKS = 6
# A file each actor appends one line per look to (play_headed --decisions); unset: no log.
DECISION_LOG_ENV = 'GENERALS_DECISION_LOG'


def decision_line(frame, actions, trace):
    """One look, roughly: game time, P(act) at this look, and no-op or the order."""
    seconds = frame/60
    acting = trace.get('act_probability')
    head = '%02d:%04.1f  p(act) %s  ' % (seconds//60, seconds % 60, '  -  ' if acting is None else '%.3f' % acting)
    if not actions:
        return head+('no-op (no placement)' if trace.get('placement_unavailable') else 'no-op')+top_orders(trace)
    order = actions[0]
    parts = [order['kind']]
    for key in ('template', 'command'):
        if isinstance(order.get(key), str):
            parts.append(order[key])
    units = order.get('units')
    parts.append('%d unit%s' % (len(units), '' if len(units) == 1 else 's') if units else '')
    if 'target' in order:
        parts.append('-> unit %s' % order['target'])
    elif 'position' in order or 'x' in order:
        parts.append('-> point')
    return head+'  '.join(part for part in parts if part)+top_orders(trace)


def top_orders(trace):
    """The look's likeliest orders (NeuralPolicy._top_orders) on a second line, or nothing."""
    top = trace.get('top')
    return '\n           top: '+',  '.join('%s %.*f%%' % (name, 1 if p < .1 else 0, 100*p) for name, p in top) if top else ''


class EventActor:
    """One instance per player, sharing immutable weights during collection.

    The actor looks every look_ticks and decides each time whether to act (notes/wait_snapshots.md).
    Training rows are every command but only a known rate of no-op moments, so the pace they imply is
    rate x P(act)/P(no-op) orders a second, and a look acts with that pace x the time since the last look
    (NeuralPolicy.decide). The core memory advances at
    rows, as in training: at an order, and at a no-op look with probability rate x time."""
    def __init__(self, policy, seed=1, deterministic=False, science=None):
        self.policy, self.rng, self.deterministic = policy, np.random.default_rng(seed), deterministic
        self.science = science if science is not None else ScienceBuyer()
        # Default to an evenly matched rated game (strong play is the goal);
        # configure_ratings persists across reset(). Training includes unrated
        # games (rated flag off) since 7 Nov 2026; models from before saw only
        # rated matches, so pass None only to a model trained with them.
        self.ratings = rating_inputs(PLAY_RATING, PLAY_RATING)
        # Strategy input z: no opener unless configure_opener names one; persists across reset().
        self.opener = None
        self.look_ticks = LOOK_TICKS
        self.apm = apm_inputs(PLAY_APM)
        log = os.environ.get(DECISION_LOG_ENV)
        self.decision_log = open(log, 'a', encoding='utf-8', buffering=1) if log else None
        if log:
            policy.explain = True  # decide adds the top orders to the trace (decision_line)
        self.reset()

    def reset(self):
        # The last committed row's frame, the last look's frame and the next look's.
        self.previous_frame = self.last_look = None
        self.next_decision_frame = 0
        self.home_position = None
        self.rate = 60
        self.enemy_identity = (0, 0)
        self.counters = Counter()
        self.last = {}
        # The policy's core LSTM state at this player's last committed row (None: game start).
        self.memory = None
        # The game's weapon/armour stats (generals_policy.TemplateTable), built from every observation.
        self.templates = None

    def configure_clock(self, rate):
        canonical_frame(0, rate)
        self.rate = rate

    def configure_enemy(self, faction):
        """The opponent's native faction name, as a lobby shows it; None or a random pick is unknown."""
        self.enemy_identity = self.policy.encoder.identity.player_identity(faction)
        self.science.configure(faction)

    def configure_ratings(self, own=PLAY_RATING, opponent=PLAY_RATING):
        """Pre-match ratings to condition on (default: an evenly matched rated game).

        None marks a side unrated (rated flag off); pass it only to a model
        trained with unrated matches (selection since 7 Nov 2026)."""
        self.ratings = rating_inputs(own, opponent)

    def configure_opener(self, templates=None):
        """The opener to follow: policy template indices of OPENER_BUILDINGS build orders
        (openers.named_templates), or None for none. Shown only in the opening window, as in training."""
        opener_inputs(self.policy.config, templates)  # validates
        self.opener = None if templates is None else list(templates)

    def configure_apm(self, apm=PLAY_APM):
        """The pace to play at, as the model's APM input (commands per minute); None is unknown."""
        self.apm = apm_inputs(apm)

    def configure_look(self, milliseconds=1000*LOOK_TICKS/60):
        """Game time between looks: reaction time and compute. 100-200 ms by design."""
        ticks = round(milliseconds*60/1000)
        if not 1 <= ticks <= 60:
            raise ValueError('The look interval must be in 17..1000 ms')
        self.look_ticks = ticks

    @property
    def next_deadline_native(self):
        return native_deadline(self.next_decision_frame, self.rate)

    def act(self, observation, terrain, frame):
        # Before any skip: a template's stats arrive once, in the observation where it first appears.
        if self.templates is None:
            self.templates = TemplateTable(self.rate)
        self.templates.add(observation.get('template_stats'))
        frame = canonical_frame(frame, self.rate)
        # Science purchases are scripted and checked on every observation, not on the model's schedule.
        bought = self.science(observation)
        self.counters['science_purchases'] += len(bought)
        if frame < self.next_decision_frame:
            self.counters['skipped_inferences'] += 1
            return bought
        return self._look(bought, observation, terrain, frame)

    def _look(self, bought, observation, terrain, frame):
        encoded = self._encode(observation, terrain, frame)
        elapsed = self.look_ticks if self.last_look is None else frame-self.last_look
        # Snapshots the training rows would hold in this time: the action rate, and the memory's commit rate.
        expected = self.policy.snapshot_rate*elapsed/60
        with no_grad():
            actions, trace, memory, value, logp, entropy = self.policy.decide(
                encoded, self.memory, rng=self.rng, deterministic=self.deterministic, snapshots=expected)
        self.last_look = frame
        self.next_decision_frame = frame + (max(self.look_ticks, MIN_ORDER_TICKS) if actions else self.look_ticks)
        if actions or self.rng.random() < expected:
            self.memory, self.previous_frame = memory, frame
            self.counters['memory_commits'] += 1
        self._decided(encoded, trace, frame, actions, value, logp, entropy)
        if self.decision_log is not None:
            self.decision_log.write(decision_line(frame, actions, trace)+'\n')
        return bought + actions

    def _encode(self, observation, terrain, frame):
        observation = normalize_demonstration({'frame': 0, 'hold_ticks': 0,
            'observation': observation}, self.rate)['observation']
        if self.home_position is None:
            self.home_position = initial_home(observation)
        encoded = self.policy.encoder.encode(observation, terrain, frame, previous_frame=self.previous_frame,
            home_position=self.home_position, source_ticks_per_second=self.rate)
        encoded['event_rate'] = self.rate
        encoded['enemy_identity'] = self.enemy_identity
        # Only a model trained with the rating input is conditioned: config rating_input (default off), or a
        # policy flagged rating_input=True by its loader (trained with GENERALS_RATING_INPUT=1, which the
        # config does not record; evaluate_events --rating-input).
        rated = self.policy.config.get('rating_input', False) or getattr(self.policy, 'rating_input', False)
        encoded['ratings'] = self.ratings if rated else UNRATED
        encoded['apm'] = self.apm
        encoded['opener'] = opener_inputs(self.policy.config, self.opener, frame/60)
        encoded.update(self.templates.arrays())
        if self.policy.config['placement_mode'] == 'full_mask':
            if not hasattr(self, 'placement_query'):
                raise ValueError('Placement-aware event actor needs a placement-capable environment')
            encoded['placement_query'] = self.placement_query
        return encoded

    def _decided(self, encoded, trace, frame, actions, value, logp, entropy):
        self.counters['decisions'] += 1
        self.counters['commands' if actions else 'defers'] += 1
        self.last = {'encoded': encoded, 'trace': trace, 'frame': frame,
                     'value': float(value.data), 'logp': float(logp.data), 'entropy': float(entropy.data)}

    def feedback(self, feedback):
        for result in feedback:
            self.counters['accepted' if result['accepted'] else 'rejected'] += 1
            if result.get('placement_adjusted'):
                self.counters['placement_adjusted'] += 1
                self.counters['placement_distance_total'] += result.get('placement_distance', 0)
