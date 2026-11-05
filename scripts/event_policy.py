"""Event policy configuration and the deadline-driven deployed actor."""
from __future__ import annotations

from collections import Counter
import numpy as np

from event_schedule import canonical_frame, native_deadline
from generals_identity import vanilla_china_production
from generals_policy import ARCHITECTURE, OBSERVATION_VERSION, PLAY_RATING, initial_home, rating_inputs
from nn_numpy import no_grad
from replay_sequences import normalize_demonstration

PLACEMENT_MODES = ('fast', 'full_mask')


def command_vocabulary(command_catalog):
    """Canonical commands, general's promotions and upgrades from the native catalogue."""
    commands = command_catalog['commands']
    return {'commands': sorted(name for name, info in commands.items() if info['kind'] in ('upgrade', 'ability')),
            'command_info': {name: [info['kind'], info['target']] for name, info in sorted(commands.items())},
            'command_menus': command_catalog['menus'], 'command_shortcuts': command_catalog['shortcuts'],
            'promotions': command_catalog['promotions'], 'upgrades': sorted(command_catalog['upgrades'])}


def default_config(vocabulary, placement_mode='fast'):
    config = {'types': vocabulary['types'], 'templates': vocabulary['templates'], 'menus': vocabulary['menus'],
              'commands': vocabulary['commands'], 'command_info': vocabulary['command_info'],
              'command_menus': vocabulary['command_menus'], 'command_shortcuts': vocabulary['command_shortcuts'],
              'promotions': vocabulary['promotions'], 'upgrades': vocabulary['upgrades'],
              'sciences': sorted({name for names in vocabulary['promotions'].values() for name in names}),
              'architecture': ARCHITECTURE, 'observation_policy_version': OBSERVATION_VERSION,
              # Widths are model-only and never change shard columns.
              'hidden': 1408, 'entity_hidden': 256, 'embedding': 64, 'attention_heads': 2,
              'map_side': 16, 'map_hidden': 128, 'scalar_hidden': 256, 'max_group': 64,
              'position_bins': 256, 'angle_bins': 16,
              'event_max_delay': 120, 'delay_ticks': list(range(1, 121)), 'event_observation_ms': 100,
              'variant_l2': 1e-4, 'delay_loss_weight': .2,
              'argument_loss_weights': {'verb': 2.0, 'unit': .5, 'target': .5,
                                        'template': 1.0, 'command': 1.0, 'science': 1.0, 'slot': 1.0,
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


class EventActor:
    """One instance per player, sharing immutable weights during collection."""
    def __init__(self, policy, seed=1, deterministic=False):
        self.policy, self.rng, self.deterministic = policy, np.random.default_rng(seed), deterministic
        self.reset()

    def reset(self):
        self.previous_frame = None
        self.next_decision_frame = 0
        self.home_position = None
        self.rate = 60
        self.enemy_identity = (0, 0)
        self.ratings = rating_inputs(PLAY_RATING, None)
        self.counters = Counter()
        self.last = {}

    def configure_clock(self, rate):
        canonical_frame(0, rate)
        self.rate = rate

    def configure_enemy(self, faction):
        """The opponent's native faction name, as a lobby shows it; None or a random pick is unknown."""
        self.enemy_identity = self.policy.encoder.identity.player_identity(faction)

    def configure_ratings(self, own=PLAY_RATING, opponent=None):
        """Pre-match ratings to condition on; None is unrated (the default opponent)."""
        self.ratings = rating_inputs(own, opponent)

    @property
    def next_deadline_native(self):
        return native_deadline(self.next_decision_frame, self.rate)

    def act(self, observation, terrain, frame):
        frame = canonical_frame(frame, self.rate)
        if frame < self.next_decision_frame:
            self.counters['skipped_inferences'] += 1
            return []
        observation = normalize_demonstration({'frame': 0, 'hold_ticks': 0,
            'observation': observation}, self.rate)['observation']
        if self.home_position is None:
            self.home_position = initial_home(observation)
        encoded = self.policy.encoder.encode(observation, terrain, frame, previous_frame=self.previous_frame,
            home_position=self.home_position, source_ticks_per_second=self.rate)
        encoded['event_rate'] = self.rate
        encoded['enemy_identity'] = self.enemy_identity
        encoded['ratings'] = self.ratings
        if self.policy.config['placement_mode'] == 'full_mask':
            if not hasattr(self, 'placement_query'):
                raise ValueError('Placement-aware event actor needs a placement-capable environment')
            encoded['placement_query'] = self.placement_query
        with no_grad():
            actions, trace, _, value, logp, entropy = self.policy.decide(
                encoded, rng=self.rng, deterministic=self.deterministic)
        self.previous_frame = frame
        self.next_decision_frame = frame + self.policy.config['delay_ticks'][trace['delay']]
        self.counters['decisions'] += 1
        self.counters['commands' if actions else 'defers'] += 1
        self.last = {'encoded': encoded, 'trace': trace, 'frame': frame,
                     'value': float(value.data), 'logp': float(logp.data), 'entropy': float(entropy.data)}
        return actions

    def feedback(self, feedback):
        for result in feedback:
            self.counters['accepted' if result['accepted'] else 'rejected'] += 1
            if result.get('placement_adjusted'):
                self.counters['placement_adjusted'] += 1
                self.counters['placement_distance_total'] += result.get('placement_distance', 0)
