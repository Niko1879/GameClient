"""Memory-free event policy: encoder, factual action masks, teacher targets and NumPy inference.

Relational entities, spatial pointers and autoregressive argument heads, inspired by
AlphaStar's modalities. The PyTorch learner (torch_policy_v3.py) trains these same
parameters; this module is the deployed inference path and the reference for tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from collections import OrderedDict
import threading

import numpy as np

from generals_identity import IdentityHierarchy, in_production_scope, validate_production_scope
from nn_numpy import Adam, Tensor, categorical, concatenate, conv2d_same, scatter_mean

ARCHITECTURE = 'generals-relational-v4-event'
OBSERVATION_VERSION = 7
KINDS = ('wait', 'move', 'attack_move', 'attack', 'stop', 'sell', 'build', 'train', 'rally', 'dock',
         'resume_construction', 'upgrade', 'science', 'ability', 'enter', 'exit', 'evacuate',
         'guard', 'guard_air', 'cancel_production', 'cancel_construction', 'scatter', 'add_waypoint',
         'force_attack', 'force_attack_ground', 'repair', 'get_repaired')
# Ability commands add an entity target or a position according to their command.
TARGET_KINDS = ('attack', 'dock', 'resume_construction', 'enter', 'force_attack', 'repair', 'get_repaired')
BUILDER_KINDS = ('build', 'resume_construction')
GUARD_KINDS = ('guard', 'guard_air')
GROUP_KINDS = ('move', 'attack_move', 'attack', 'stop', 'sell', 'dock', 'ability', 'enter', 'exit', 'evacuate',
               'cancel_construction', 'scatter', 'add_waypoint', 'force_attack', 'force_attack_ground',
               'repair', 'get_repaired') + GUARD_KINDS
SPATIAL_KINDS = ('move', 'attack_move', 'build', 'rally', 'add_waypoint', 'force_attack_ground') + GUARD_KINDS
# Native build-queue buttons; cancel_production chooses one occupied slot.
QUEUE_SLOTS = 9
# Owned actor/observable target pairs exported by the engine for each kind.
PAIR_FIELDS = {'dock': 'dock_targets', 'enter': 'enter_targets', 'repair': 'repair_targets',
               'get_repaired': 'repair_depots'}
COMMAND_KINDS = ('upgrade', 'ability')
ENTITY_FEATURES = 14
EXTRA_ENTITY_FEATURES = 14
ENTITY_UPGRADE_SLOTS = 4
SCALAR_FEATURES = 16
MAP_CHANNELS = 12
RELATIVE_FEATURES = 5


def selection_limit(config, kind):
    return 1 if kind not in GROUP_KINDS or kind in ('exit', 'cancel_construction') else config['max_group']


def command_target(config, target):
    """'entity', 'position' or None for a teacher/decision target dictionary."""
    kind = target['kind']
    if kind in TARGET_KINDS:
        return 'entity'
    if kind in SPATIAL_KINDS:
        return 'position'
    if kind == 'ability':
        mode = config['command_info'][config['commands'][target['command']]][1]
        return {'object': 'entity', 'position': 'position'}.get(mode)
    return None


class Unrepresentable(ValueError):
    pass


def entity_key(entity):
    # IDs are only used to resolve commands. They are never neural features.
    return (not entity['own'], not entity['visible'], entity['owner'] < 2, entity['type'],
            tuple(entity['position']), entity['last_seen_frame'])


def initial_home(observation):
    entities = observation['own_entities']
    return np.mean([e['position'][:2] for e in entities], axis=0).tolist() if entities else None


# Rows whose actor masks stay cached: above one shard chunk (128 rows).
MASK_CACHE_ROWS = 160


class Encoder:
    def __init__(self, config):
        self.config = config
        self.types = {name: i for i, name in enumerate(config['types'])}
        self.identity = IdentityHierarchy(config)
        self.upgrade_ids = {name: i+1 for i, name in enumerate(config['upgrades'])}
        # Per-game inputs, found by the identity of the game's terrain record.
        self._extent, self._static, self._player, self._home = None, {}, {}, None
        self._menus = {}

    def _bounds(self, terrain):
        extent = self._extent
        if extent is None or extent[0] is not terrain:
            if terrain['grid_size'] != 32:
                raise ValueError('Observation schema expects a 32x32 map')
            low, high = np.asarray(terrain['extent_low'][:2]), np.asarray(terrain['extent_high'][:2])
            span = high - low
            if np.any(span <= 0):
                raise ValueError('Invalid map extent')
            extent = self._extent = (terrain, low, high, span)
        return extent[1:]

    def _static_map(self, terrain, low, high, span, rotated):
        """Map columns that do not change during a game, in one orientation."""
        entry = self._static.get(rotated)
        if entry is not None and entry[0] is terrain:
            return entry[1], entry[2]
        n, scale = self.config['map_side'], 32 // self.config['map_side']
        height, cliff = np.asarray(terrain['height']), np.asarray(terrain['cliff'])
        sources = terrain.get('supply_sources', [])
        if rotated:
            height, cliff = (grid.reshape(32, 32)[::-1, ::-1].reshape(-1) for grid in (height, cliff))
            sources = [[*(low+high-np.asarray(s['position'][:2])).tolist(), *s['position'][2:]] for s in sources]
        else:
            sources = [s['position'] for s in sources]
        def pool(values):
            return values.reshape(n, scale, n, scale).mean(axis=(1, 3)).reshape(-1)
        coords = (np.stack(np.meshgrid(np.arange(n), np.arange(n)), axis=-1).reshape(-1, 2) + .5) / n
        base = np.zeros((n*n, MAP_CHANNELS))
        base[:, 0] = pool(height) / 512
        base[:, 1] = pool(cliff)
        base[:, 10:12] = coords
        # Public static supply positions are available before scouting. Never
        # import supply quantities or enemy entities from the global simulator.
        for source in sources:
            pos = (np.asarray(source[:2])-low)/span
            x, y = np.clip((pos*n).astype(int), 0, n-1)
            base[y*n+x, 9] += 1
        self._static[rotated] = (terrain, base, coords)
        return base, coords

    def _player_flags(self, player):
        """Completed player upgrades, owned promotions and currently purchasable promotions."""
        key = (tuple(player.get('upgrades', [])), tuple(player.get('sciences', [])),
               tuple(player.get('purchasable_sciences', [])))
        flags = self._player.get(key)
        if flags is None:
            upgrades, sciences = self.config['upgrades'], self.config['sciences']
            owned_upgrades, owned_sciences, purchasable = map(set, key)
            if len(self._player) > 4096:
                self._player = {}
            flags = self._player[key] = (
                np.asarray([name in owned_upgrades for name in upgrades]
                           + [name in owned_sciences for name in sciences]
                           + [name in purchasable for name in sciences], dtype=float),
                np.asarray([name in purchasable for name in sciences], dtype=bool))
        return flags

    def _menu_flags(self, name):
        flags = self._menus.get(name)
        if flags is None:
            menus = self.config['menus']
            flags = self._menus[name] = (bool(menus.get(name+'|build')), bool(menus.get(name+'|train')))
        return flags

    def schedule_inputs(self, observation, terrain):
        """What policy.target() reads to accept or reject a command.

        Acceptance does not depend on entity order or map orientation, so the
        capture schedule skips them and every feature column (test_scheduled_capture).
        """
        low, _, span = self._bounds(terrain)
        entities = observation['own_entities'] + observation['known_nonowned_entities']
        real_count = len(entities)
        return {'entities': entities, 'ids': {e['id']: i for i, e in enumerate(entities)},
                'own': np.asarray([e['own'] for e in entities] if real_count else [False]),
                'visible_target': np.asarray([not e['own'] and e['visible'] and e['health'] > 0 and e['max_health'] > 0
                                              for e in entities] if real_count else [False]),
                'low': low, 'span': span, 'rotated': False,
                'purchasable_sciences': self._player_flags(observation['player'])[1]}

    def encode(self, observation, terrain, frame, *, previous_frame=None, home_position=None,
               source_ticks_per_second=60):
        if observation.get('observation_policy_version') != OBSERVATION_VERSION:
            raise ValueError('Policy requires observation version %d' % OBSERVATION_VERSION)
        if observation['player']['faction'] != 'FactionChina':
            raise ValueError('Policy is for vanilla China')
        low, high, span = self._bounds(terrain)
        # Canonical coordinates: every player starts in the same half of the map.
        home_position = home_position if home_position is not None else initial_home(observation)
        cached = self._home
        if cached is not None and cached[0] is home_position and home_position is not None and cached[1] is low:
            rotated = cached[2]
        else:
            home = (np.asarray(home_position)-low)/span-.5 if home_position is not None else np.zeros(2)
            rotated = bool(home[int(np.argmax(abs(home)))] > 0)
            self._home = (home_position, low, rotated)
        n = self.config['map_side']
        if rotated:
            def rotate(rows):
                # The mirrored position for every row at once: the same float64 operations.
                if not rows:
                    return []
                xy = (low+high-np.asarray([e['position'][:2] for e in rows], dtype=float)).tolist()
                return [{**e, 'position': [*point, *e['position'][2:]]} for e, point in zip(rows, xy)]
            observation = {**observation,
                           'own_entities': rotate(observation['own_entities']),
                           'known_nonowned_entities': rotate(observation['known_nonowned_entities']),
                           'visibility_grid': np.asarray(observation['visibility_grid']).reshape(32, 32)[::-1, ::-1].reshape(-1)}
        entities = sorted(observation['own_entities'] + observation['known_nonowned_entities'], key=entity_key)
        real_count = len(entities)
        types = [self.types.get(entity['type'], 0) for entity in entities]
        owns = [e['own'] for e in entities]
        visibles = [e['visible'] for e in entities]
        if real_count:
            # Column-wise, with the same float64 operations as a per-entity row.
            def column(name, default=None):
                return np.asarray([e[name] if default is None else e.get(name, default) for e in entities], dtype=float)
            position = np.asarray([e['position'][:3] for e in entities], dtype=float)
            xy = (position[:, :2] - low) / span
            health, maximum = column('health'), column('max_health')
            construction = column('construction_percent', -1)
            queues = [e.get('production_queue', []) for e in entities]
            numeric = np.column_stack([
                xy[:, 0], xy[:, 1], position[:, 2] / 512,
                health / np.maximum(maximum, 1),
                np.log1p(np.maximum(maximum, 0)) / 10,
                np.asarray(owns, dtype=float), np.asarray(visibles, dtype=float),
                np.asarray([e['owner'] >= 2 and not e['own'] for e in entities], dtype=float),
                np.log1p(np.maximum(0, frame - column('last_seen_frame')) / 60) / 10,
                (construction == -1).astype(float),
                np.maximum(0, construction) / 100,
                np.asarray([min(len(q), 20) / 20 for q in queues], dtype=float),
                np.asarray([np.mean([item['progress'] for item in q]) / 100 if q else 0 for q in queues], dtype=float),
                np.asarray([min(sum(item['remaining_quantity'] for item in q), 20) / 20 for q in queues], dtype=float)])
        else:
            # A zero row keeps pooling and pointer shapes defined after defeat.
            numeric = np.asarray([[0] * ENTITY_FEATURES])
        p = observation['player']
        since_decision = max(0, frame - (frame if previous_frame is None else previous_frame))
        submission = observation.get('last_submission', {})
        since_command = submission.get('age_seconds', 0) if submission.get('available', False) else 0
        scalars = np.concatenate([np.minimum(np.maximum(np.asarray([
            np.log1p(p['cash']) / 10, p['power_production'] / 100, p['power_consumption'] / 100,
            float(p['radar']), p['rank'] / 5, p['skill_points'] / 10, p['science_points'] / 10,
            float(p['dead']), np.log1p(frame / 60) / 10, np.log1p(since_decision / 60),
            len(observation['own_entities']) / 100, len(observation['known_nonowned_entities']) / 100,
            span[0] / 4000, span[1] / 4000]), -10), 10),
            [np.log1p(since_command)/5, source_ticks_per_second/60]])
        visible_target = np.asarray([not e['own'] and e['visible'] and e['health'] > 0 and e['max_health'] > 0
                                     for e in entities] if real_count else [False])
        encoded = {'entities': entities, 'ids': {e['id']: i for i, e in enumerate(entities)},
                   'numeric': numeric, 'types': np.asarray(types if types else [0], dtype=int),
                   'own': np.asarray(owns if real_count else [False]),
                   'visible_target': visible_target, 'scalars': scalars,
                   'low': low, 'span': span, 'rotated': rotated, 'frame': frame}
        if self.config['placement_mode'] == 'full_mask':
            encoded['placement_records'] = observation.get('build_placements', [])
            encoded['placement_source_rate'] = source_ticks_per_second
        # Map cells retain categorical shroud information. The native enum is
        # 0=clear, 1=fogged, 2=unseen, not a continuous visibility score.
        visibility = np.asarray(observation['visibility_grid']).reshape(32, 32)
        scale = 32 // n
        base, coords = self._static_map(terrain, low, high, span, rotated)
        map_inputs = base.copy()
        for category in range(3):
            map_inputs[:, 2 + category] = np.add.reduce((visibility == category).reshape(n, scale, n, scale),
                                                        axis=(1, 3), dtype=float).reshape(-1) / (scale*scale)
        upgrade_ids, catalog_types = self.upgrade_ids, self.types
        entity_upgrades = np.zeros((max(real_count, 1), ENTITY_UPGRADE_SLOTS), dtype=int)
        info = self.config['command_info']
        grid_xy = np.clip((numeric[:real_count, :2] * n).astype(int), 0, n-1)
        cells = (grid_xy[:, 1]*n+grid_xy[:, 0]).tolist()
        channels = [5 if own else (6 if visible and e['owner'] >= 2 else 7 if e['owner'] >= 2 else 8)
                    for e, own, visible in zip(entities, owns, visibles)]
        queued_types = [0]*real_count
        extra_rows, extra_index = [], []
        for index, e in enumerate(entities):
            if not e['own']:
                # Owner-only command capabilities; nonowned rows stay zero.
                if 'occupants' in e or 'buildable_templates' in e or 'trainable_templates' in e:
                    extra_index.append(index)
                    extra_rows.append([False, False, min(len(e.get('buildable_templates', [])), 20)/20,
                                       min(len(e.get('trainable_templates', [])), 20)/20,
                                       0., 0., 0., 0., 0., 0, 0, 0., e.get('occupants', 0), 0])
                continue
            has_build, has_train = self._menu_flags(e['type'])
            commands = e.get('commands', [])
            kinds = [info.get(name, ('', ''))[0] for name in commands]
            charging = e.get('command_cooldowns', [])
            capacity = e.get('capacity', 0)
            extra_index.append(index)
            extra_rows.append([has_build, has_train,
                               min(len(e.get('buildable_templates', [])), 20)/20,
                               min(len(e.get('trainable_templates', [])), 20)/20,
                               float('ability' in kinds), float('upgrade' in kinds), float('evacuate' in kinds),
                               min(len(commands), 10)/10, float(bool(charging)),
                               min(c['ready_percent'] for c in charging)/100 if charging else 0,
                               float(bool(e.get('container_id', 0))), float(capacity > 0),
                               e.get('occupants', 0)/max(capacity, 1), float(bool(e.get('enter_targets')))])
            names = e.get('upgrades')
            if names:
                known = [upgrade_ids[name] for name in names if name in upgrade_ids]
                entity_upgrades[index, :len(known[:ENTITY_UPGRADE_SLOTS])] = known[:ENTITY_UPGRADE_SLOTS]
            queue = e.get('production_queue')
            if queue:
                queued_types[index] = catalog_types.get(queue[0].get('unit_type'), 0)
        if real_count:
            extra = np.zeros((real_count, EXTRA_ENTITY_FEATURES))
            if extra_rows:
                extra[extra_index] = extra_rows
        else:
            extra = np.asarray([[0]*EXTRA_ENTITY_FEATURES])
        if cells:  # entity counts per cell and category (exact in any order)
            np.add.at(map_inputs, (np.asarray(cells), np.asarray(channels)), 1)
        map_inputs[:, 5:10] = np.log1p(map_inputs[:, 5:10]) / 4
        player_flags, purchasable_sciences = self._player_flags(p)
        encoded.update(map_inputs=map_inputs, map_coords=coords, cells=np.asarray(cells or [0]),
                       extra_numeric=extra,
                       queued_types=np.asarray(queued_types or [0]),
                       entity_upgrades=entity_upgrades, player_flags=player_flags,
                       purchasable_sciences=purchasable_sciences,
                       player_identity=self.identity.player_identity(p['faction']))
        from unit_tasks import encode
        # Unit-task coordinates remain native world coordinates in the
        # transport snapshot; this helper rotates them exactly once.
        encode(self.config, observation, encoded, rotated=rotated, types=self.types)
        return encoded


class NeuralPolicy:
    def __init__(self, config, seed=1, *, parameters=True):
        """parameters=False builds the encoder, masks and targets only (preparation and identities)."""
        if config.get('architecture') != ARCHITECTURE:
            raise ValueError('Unsupported policy architecture')
        validate_production_scope(config)
        self.config, self.encoder = config, Encoder(config)
        self.kinds = KINDS
        self._masks = threading.local()
        self.command_index = {name: i for i, name in enumerate(config['commands'])}
        self.science_index = {name: i for i, name in enumerate(config['sciences'])}
        self._choices, self._menus = set(config['templates']), {}
        self.parameters = {}
        self.mobile_types, self.structure_types = set(), set()
        for key, choices in config['menus'].items():
            name, operation = key.rsplit('|', 1)
            if operation == 'train':
                self.mobile_types.update(choices)
                self.structure_types.add(name)
            else:
                self.mobile_types.add(name)
                self.structure_types.update(choices)
        self.structure_types -= self.mobile_types
        n, bins = config['map_side'], config['position_bins']
        if n <= 0 or 32 % n or bins % n or config['entity_hidden'] % config['attention_heads']:
            raise ValueError('Invalid map/attention dimensions')
        if config['delay_ticks'] != list(range(1, config['event_max_delay']+1)):
            raise ValueError('Unsupported delay vocabulary')
        if not np.isfinite(config.get('variant_l2', 0)) or config.get('variant_l2', 0) < 0:
            raise ValueError('Variant regularization must be finite and nonnegative')
        if config['placement_mode'] == 'full_mask' and config['angle_bins'] != 16:
            raise ValueError('Build angles must match the native 16-bin vocabulary')
        if not parameters:
            return
        rng = np.random.default_rng(seed)
        identity_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        h, d, e, m, s = (config[k] for k in ('hidden', 'entity_hidden', 'embedding', 'map_hidden', 'scalar_hidden'))
        def w(name, shape, gain=1, *, generator=None):
            self.parameters[name] = Tensor((generator or rng).normal(0, gain/math.sqrt(shape[0]), shape), requires_grad=True)
        def b(name, count, value=0):
            self.parameters[name] = Tensor(np.full(count, value), requires_grad=True)
        vocab = self.encoder.identity.config
        # Embedding scale depends on vector width, not vocabulary size.
        self.parameters['family_embedding'] = Tensor(
            identity_rng.normal(0, 1/math.sqrt(e), (len(vocab['families']), e)), requires_grad=True)
        self.parameters['base_faction_embedding'] = Tensor(
            identity_rng.normal(0, .25/math.sqrt(e), (len(vocab['base_factions']), e)), requires_grad=True)
        self.parameters['base_faction_embedding'].data[0] = 0
        b('general_embedding', (len(vocab['generals']), e))
        b('variant_embedding', (len(vocab['objects']), e))
        w('entity_w', (ENTITY_FEATURES+EXTRA_ENTITY_FEATURES+3*e, d)); b('entity_b', d)
        # Row 0 is the empty upgrade slot; it is masked out of every sum.
        w('upgrade_embedding', (len(config['upgrades'])+1, e), .5)
        self.parameters['upgrade_embedding'].data[0] = 0
        w('player_flags_w', (len(config['upgrades'])+2*len(config['sciences']), s), .2)
        w('command_embedding', (len(config['commands']), e), 1)
        w('command_query_w', (h, e), .1); b('command_b', len(config['commands']))
        w('command_context_w', (e, h), .2)
        w('science_embedding', (len(config['sciences']), e), 1)
        w('science_query_w', (h, e), .1); b('science_b', len(config['sciences']))
        w('science_context_w', (e, h), .2)
        w('slot_w', (h, QUEUE_SLOTS), .1); b('slot_b', QUEUE_SLOTS)
        for name in ('attn_q', 'attn_k', 'attn_v', 'attn_out', 'entity_ff'):
            w(name, (d, d), .5)
        w('relation_bias', (4, config['attention_heads']), .1)
        w('scatter_w', (d, 8), .5)
        w('map_input_w', (MAP_CHANNELS+8, m)); b('map_input_b', m)
        w('map_conv', (9*m, m), .5); b('map_conv_b', m)
        w('scalar_w', (SCALAR_FEATURES, s)); b('scalar_b', s)
        w('player_identity_w', (e, s), .5, generator=identity_rng)
        from unit_tasks import NUMERIC, RELATIONS, RELATION_FEATURES, COMMAND_FEATURES, VALIDATION_COLUMNS
        w('task_w', (NUMERIC, d), .5)
        self.parameters['task_w'].data[list(VALIDATION_COLUMNS)] = 0
        for role in RELATIONS:
            w('task_'+role+'_w', (d, d), .25)
            w('task_'+role+'_features_w', (RELATION_FEATURES, d), .25)
        self.parameters['task_validation_target_w'].data[:] = 0
        self.parameters['task_validation_target_features_w'].data[:] = 0
        w('task_command_w', (COMMAND_FEATURES, s), .5)
        w('task_command_template_w', (e, s), .25)
        b('task_validation_w', (COMMAND_FEATURES, s))
        b('task_validation_template_w', (e, s))
        if config['placement_mode'] == 'full_mask':
            w('build_map_w', (1, m), .2)
            w('build_angle_embedding', (16, h), .2)
        w('event_input_w', (s+4*d+m, h), .7); b('event_input_b', h)
        w('event_hidden_w', (h, h), .7); b('event_hidden_b', h)
        w('pool_query', (s, d), .2)
        w('map_pool_query', (s, m), .2)
        w('verb_w', (h, len(KINDS)), .1); b('verb_b', len(KINDS))
        w('verb_embedding', (len(KINDS), h), .2)
        w('context_w', (h, h), .5); w('selected_w', (d, h), .5)
        w('actor_pointer_w', (h, d), .2); w('target_pointer_w', (h, d), .2)
        w('stop_w', (h, 1), .1); b('stop_b', 1, 1)
        b('template_b', len(config['templates']))
        # The same object representation informs sightings, queues, choice
        # logits and placement context. Exact template IDs still dispatch.
        w('template_query_w', (h, e), .1, generator=identity_rng)
        w('template_context_w', (e, h), .2, generator=identity_rng)
        w('spatial_query', (h, m), .2)
        # Relative cell positions to the selected units are learned features,
        # not a rule selecting attack/defend/harass sites.
        w('relative_w', (RELATIVE_FEATURES, m), .5)
        w('cell_context_w', (m, h), .5)
        w('offset_w', (h, (bins//n)**2), .1); b('offset_b', (bins//n)**2)
        w('angle_w', (h, config['angle_bins']), .1); b('angle_b', config['angle_bins'])
        w('delay_w', (h, len(config['delay_ticks'])), .1); b('delay_b', len(config['delay_ticks']))
        w('value_hidden_w', (h, s), .5); b('value_hidden_b', s)
        w('value_w', (s, 1), .1); b('value_b', 1)

    def object_embedding(self, objects):
        """Shared family/base faction plus zero-initialized general/type residuals."""
        identity, p = self.encoder.identity, self.parameters
        return (p['family_embedding'][identity.families[objects]]
                + p['base_faction_embedding'][identity.base_factions[objects]]
                + p['general_embedding'][identity.generals[objects]]
                + p['variant_embedding'][objects])

    def type_embedding(self, types):
        return self.object_embedding(self.encoder.identity.type_objects[types])

    def regularization_loss(self):
        if not self.config.get('variant_l2', 0):
            return Tensor(0)
        # Apply once per optimizer update. There is no shrinkage on the shared
        # families or base faction representations.
        loss = Tensor(0)
        for name in ('general_embedding', 'variant_embedding', 'template_b'):
            parameter = self.parameters[name]
            loss = loss + (parameter*parameter).sum()
        return loss*self.config['variant_l2']

    def actor_mask(self, encoded, kind):
        """Owned actors that can perform `kind` now (a fresh copy; callers may modify it).

        Masks depend only on one encoded observation. Preparation asks for each
        kind several times per row (verb mask and target, then legality when a
        chunk of up to 128 rows is written), so recent observations' masks are
        kept per thread. Entries hold their observation, so ids are not reused.
        """
        return self._cached_mask(encoded, kind)[0].copy()

    def _cached_mask(self, encoded, kind):
        """(mask, whether any actor passes): the caller must not modify the mask."""
        cache = getattr(self._masks, 'recent', None)
        if cache is None:
            cache = self._masks.recent = OrderedDict()
        entry = cache.get(id(encoded))
        if entry is None or entry[0] is not encoded:
            rows = [(i, e) for i, e in enumerate(encoded['entities']) if e['own']]
            entry = cache[id(encoded)] = (encoded, {}, rows)
            if len(cache) > MASK_CACHE_ROWS:
                cache.popitem(last=False)
        masks = entry[1]
        cached = masks.get(kind)
        if cached is None:
            mask = self._actor_mask(encoded, kind, entry[2])
            cached = masks[kind] = (mask, bool(mask.any()))
        return cached

    def _production_menu(self, name, kind):
        """The templates this type offers for `kind`, within the production scope."""
        key = (name, kind)
        menu = self._menus.get(key)
        if menu is None:
            menu = self.config['menus'].get(name + '|' + ('train' if kind == 'rally' else kind), [])
            if kind in ('build', 'train') and self.config.get('production_scope') is not None:
                choices = self._choices
                menu = [template for template in menu if template in choices] if in_production_scope(self.config, name) else []
            menu = self._menus[key] = frozenset(menu)
        return menu

    def _actor_mask(self, encoded, kind, rows):
        """`rows` lists the owned entities: every other mask entry starts and stays False."""
        mask = encoded['own'].copy()
        if kind in PAIR_FIELDS:
            mask[[i for i, e in rows if not self._has_pair(encoded, i, e, kind)]] = False
        elif kind == 'resume_construction':
            entities, ids = encoded['entities'], encoded['ids']
            def resumes(i, actor):
                if actor.get('health', 0) <= 0 or actor.get('construction_percent', -1) != -1:
                    return False
                for identity in actor['resume_targets']:
                    j = ids.get(identity)
                    if j is not None and i != j:
                        target = entities[j]
                        if target['own'] and target.get('health', 0) > 0 and target.get('construction_percent', -1) >= 0:
                            return True
                return False
            mask[[i for i, e in rows if not resumes(i, e)]] = False
        elif kind in ('build', 'train', 'rally'):
            field = 'buildable_templates' if kind == 'build' else 'trainable_templates'
            for i, e in rows:
                menu = self._production_menu(e['type'], kind)
                passes = bool(menu and (kind == 'rally' or e.get('construction_percent') == -1))
                if kind != 'rally' and passes:
                    passes = not menu.isdisjoint(e[field])
                mask[i] = passes
        elif kind in ('move', 'attack_move', 'add_waypoint'):
            mobile = self.mobile_types
            mask[[i for i, e in rows if e['type'] not in mobile]] = False
        elif kind == 'sell':
            structure = self.structure_types
            mask[[i for i, e in rows if e['type'] not in structure]] = False
        elif kind in COMMAND_KINDS:
            mask[[i for i, e in rows if not (e.get('commands') and self._commands(e, kind))]] = False
        elif kind == 'exit':
            observable = encoded['ids']
            mask[[i for i, e in rows if e.get('container_id', 0) not in observable]] = False
        elif kind == 'evacuate':
            info = self.config['command_info']
            mask[[i for i, e in rows if not any(info.get(name, [''])[0] == 'evacuate' for name in e.get('commands', ()))]] = False
        elif kind == 'science':
            mask[:] = False
        elif kind in GUARD_KINDS or kind == 'scatter':
            # Owner-only native flag: the group dispatch reaches every mobile AI.
            mask[[i for i, e in rows if not e.get('can_guard', False)]] = False
        elif kind == 'cancel_production':
            mask[[i for i, e in rows if not (e.get('construction_percent', -1) == -1 and e.get('production_queue'))]] = False
        elif kind == 'cancel_construction':
            mask[[i for i, e in rows if not (e['health'] > 0 and e.get('construction_percent', -1) >= 0)]] = False
        if kind in ('attack', 'attack_move', 'force_attack', 'force_attack_ground'):
            # Owner-only engine flags: garrisons and disabled/deploying weapons need more than a type.
            moving = kind == 'attack_move'
            mask[[i for i, e in rows if not (e['can_attack'] and (not moving or e['can_attack_move'])
                                             and e['health'] > 0 and e.get('construction_percent', -1) == -1)]] = False
        return mask

    def _has_pair(self, encoded, i, actor, kind):
        """Whether this owned actor has a target in dock_matrix or pair_matrix."""
        entities, ids = encoded['entities'], encoded['ids']
        if kind == 'dock':
            if actor['health'] <= 0 or actor.get('construction_percent', -1) != -1:
                return False
            for identity in actor['dock_targets']:
                j = ids.get(identity)
                if j is not None and j != i:
                    e = entities[j]
                    if ((e['own'] or e['visible']) and e.get('construction_percent', -1) == -1
                            and not (e['max_health'] > 0 and e['health'] <= 0)):
                        return True
            return False
        if actor['health'] <= 0:
            return False
        for identity in actor.get(PAIR_FIELDS[kind], ()):
            j = ids.get(identity)
            if j is not None and j != i and (entities[j]['own'] or entities[j]['visible']):
                return True
        return False

    def dock_matrix(self, encoded):
        """Owned actor/observable target pairs from the engine's own docking legality."""
        entities = encoded['entities']
        pairs = np.zeros((len(encoded['own']), len(encoded['own'])), dtype=bool)
        observable = np.asarray([(e['own'] or e['visible'])
            and e.get('construction_percent', -1) == -1
            and not (e['max_health'] > 0 and e['health'] <= 0) for e in entities] or [False])
        for i, actor in enumerate(entities):
            if not actor['own'] or actor['health'] <= 0 or actor.get('construction_percent', -1) != -1:
                continue
            for target_id in actor['dock_targets']:
                j = encoded['ids'].get(target_id)
                if j is not None:
                    pairs[i, j] = observable[j]
            pairs[i, i] = False
        return pairs

    def _commands(self, entity, kind):
        """Owned, currently usable commands of one kind, as policy vocabulary indices."""
        if not entity['own'] or entity.get('health', 0) <= 0 or entity.get('construction_percent', -1) != -1:
            return set()
        info, index = self.config['command_info'], self.command_index
        return {index[name] for name in entity.get('commands', ()) if name in index and info[name][0] == kind}

    def command_mask(self, encoded, kind, selected):
        """Commands every selected actor can use now (the native group dispatch)."""
        mask = np.zeros(len(self.config['commands']), dtype=bool)
        if selected:
            common = set.intersection(*(self._commands(encoded['entities'][i], kind) for i in selected))
            mask[sorted(common)] = True
        return mask

    def science_mask(self, encoded):
        return np.asarray(encoded['purchasable_sciences'], dtype=bool).reshape(len(self.config['sciences']))

    def enter_matrix(self, encoded):
        """Owned actor/observable container pairs from the engine's own enter legality."""
        return self.pair_matrix(encoded, 'enter')

    def pair_matrix(self, encoded, kind):
        """Owned actor/observable target pairs from a native owner-only legality list."""
        entities, field = encoded['entities'], PAIR_FIELDS[kind]
        pairs = np.zeros((len(encoded['own']), len(encoded['own'])), dtype=bool)
        for i, actor in enumerate(entities):
            if not actor['own'] or actor['health'] <= 0:
                continue
            for target_id in actor.get(field, ()):
                j = encoded['ids'].get(target_id)
                if j is not None and j != i and (entities[j]['own'] or entities[j]['visible']):
                    pairs[i, j] = True
        return pairs

    def slot_mask(self, encoded, selected):
        mask = np.zeros(QUEUE_SLOTS, dtype=bool)
        if len(selected) == 1:
            mask[:min(len(encoded['entities'][selected[0]].get('production_queue', ())), QUEUE_SLOTS)] = True
        return mask

    def force_targets(self, encoded, selected):
        """Any currently observable live object other than the attackers themselves."""
        mask = np.asarray([(e['own'] or e['visible']) and not (e['max_health'] > 0 and e['health'] <= 0)
                           for e in encoded['entities']] or [False])
        mask[list(selected)] = False
        return mask

    def command_targets(self, encoded, selected, command):
        """Observable objects every selected actor may target with this command."""
        name = self.config['commands'][command]
        mask = np.zeros(len(encoded['own']), dtype=bool)
        sets = []
        for i in selected:
            entry = next((t for t in encoded['entities'][i].get('command_targets', ()) if t['command'] == name), None)
            sets.append(set(entry['targets']) if entry else set())
        for identity in set.intersection(*sets) if sets else ():
            j = encoded['ids'].get(identity)
            if j is not None and (encoded['entities'][j]['own'] or encoded['entities'][j]['visible']):
                mask[j] = True
        return mask

    def resume_matrix(self, encoded):
        """Exact native owner-only legality."""
        pairs = np.zeros((len(encoded['own']), len(encoded['own'])), dtype=bool)
        for i, actor in enumerate(encoded['entities']):
            if not actor['own'] or actor.get('health', 0) <= 0 or actor.get('construction_percent', -1) != -1:
                continue
            for identity in actor['resume_targets']:
                j = encoded['ids'].get(identity)
                if j is None or i == j:
                    continue
                target = encoded['entities'][j]
                pairs[i, j] = bool(target['own'] and target.get('health', 0) > 0
                    and target.get('construction_percent', -1) >= 0)
        return pairs

    def target_mask(self, encoded, kind, selected, command=None):
        if kind == 'attack':
            return encoded['visible_target']
        if kind == 'dock':
            return self.dock_matrix(encoded)[selected].all(axis=0) if selected else np.zeros_like(encoded['own'])
        if kind == 'resume_construction':
            return self.resume_matrix(encoded)[selected].all(axis=0) if len(selected) == 1 else np.zeros_like(encoded['own'])
        if kind in PAIR_FIELDS and kind != 'dock':
            return self.pair_matrix(encoded, kind)[selected].all(axis=0) if selected else np.zeros_like(encoded['own'])
        if kind == 'force_attack':
            return self.force_targets(encoded, selected)
        if kind == 'ability' and command is not None:
            return self.command_targets(encoded, selected, command)
        raise ValueError('Command has no entity target')

    def selection_mask(self, encoded, kind, selected, actors=None):
        mask = self.actor_mask(encoded, kind) if actors is None else actors.copy()
        if kind in PAIR_FIELDS and selected:
            pairs = self.dock_matrix(encoded) if kind == 'dock' else self.pair_matrix(encoded, kind)
            mask &= (pairs & pairs[selected].all(axis=0)).any(axis=1)
        if kind == 'ability' and selected:
            # Group dispatch needs one command shared by every member.
            common = set.intersection(*(self._commands(encoded['entities'][i], kind) for i in selected))
            mask &= np.asarray([bool(common & self._commands(e, kind)) for e in encoded['entities']] or [False])
        return mask

    def verb_mask(self, encoded):
        mask = np.asarray([True] + [self._cached_mask(encoded, k)[1] for k in KINDS[1:]])
        mask[KINDS.index('attack')] &= bool(encoded['visible_target'].any())
        mask[KINDS.index('science')] = bool(self.science_mask(encoded).any())
        # A forced attack needs some other observable object to aim at.
        mask[KINDS.index('force_attack')] &= int(self.force_targets(encoded, []).sum()) > 1
        return mask

    def target(self, encoded, actions):
        if not actions:
            return {'kind': 'wait'}
        if len(actions) != 1 or actions[0]['kind'] not in KINDS:
            raise Unrepresentable('Only one supported command per decision; resignation is excluded')
        a, target = actions[0], {'kind': actions[0]['kind']}
        kind = a['kind']
        config = self.config
        try:
            if not self.verb_mask(encoded)[KINDS.index(kind)]:
                raise Unrepresentable('Verb outside current action mask')
            if kind == 'science':
                target.update(selection=[], science=self.science_index[a['science']])
                if not self.science_mask(encoded)[target['science']]:
                    raise Unrepresentable('Promotion outside purchasable sciences')
                return target
            ids = a['units'] if kind in GROUP_KINDS else [a.get('builder', a.get('producer'))]
            selection = sorted({encoded['ids'][i] for i in ids})
            if not selection or len(selection) > selection_limit(config, kind):
                raise Unrepresentable('Invalid or oversized selected group')
            if kind == 'resume_construction' and len(selection) != 1:
                raise Unrepresentable('Resume construction requires one chosen builder')
            if not all(self.actor_mask(encoded, kind)[i] for i in selection):
                raise Unrepresentable('Actor outside structural action mask')
            target['selection'] = selection
            if kind in COMMAND_KINDS:
                if a['command'] not in self.command_index:
                    raise Unrepresentable('Command outside policy command scope')
                target['command'] = self.command_index[a['command']]
                if not self.command_mask(encoded, kind, selection)[target['command']]:
                    raise Unrepresentable('Command not usable by every selected actor')
            if command_target(config, target) == 'entity':
                target['target'] = encoded['ids'][a['target']]
                if not self.target_mask(encoded, kind, selection, target.get('command'))[target['target']]:
                    raise Unrepresentable('Target outside observable compatible entities')
            if kind == 'cancel_production':
                target['slot'] = a['slot']
                if type(a['slot']) is not int or not 0 <= a['slot'] < QUEUE_SLOTS or not self.slot_mask(encoded, selection)[a['slot']]:
                    raise Unrepresentable('Cancelled queue slot is empty')
            if kind in ('build', 'train'):
                target['template'] = config['templates'].index(a['template'])
                actor = encoded['entities'][selection[0]]
                if a['template'] not in config['menus'].get(actor['type'] + '|' + kind, []):
                    raise Unrepresentable('Template outside producer menu')
                if a['template'] not in actor['buildable_templates' if kind == 'build' else 'trainable_templates']:
                    raise Unrepresentable('Template outside current capabilities')
            if command_target(config, target) == 'position':
                bins = config['position_bins']
                normalized = (np.asarray(a['position'][:2]) - encoded['low']) / encoded['span']
                if np.any(normalized < 0) or np.any(normalized > 1):
                    raise Unrepresentable('Position outside map')
                x, y = np.clip((normalized * bins).astype(int), 0, bins - 1).tolist()
                # Quantize once in world coordinates; then mirror the grid index.
                if encoded['rotated']:
                    x, y = bins-1-x, bins-1-y
                n, fine = config['map_side'], bins//config['map_side']
                target.update(x=x, y=y, cell=(y//fine)*n + x//fine, offset=(y%fine)*fine + x%fine)
            if kind == 'build':
                target['angle'] = (round(a.get('angle', 0) / math.tau * config['angle_bins'])
                                   + (config['angle_bins']//2 if encoded['rotated'] else 0)) % config['angle_bins']
        except Unrepresentable:
            raise
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise Unrepresentable(str(error)) from error
        if kind == 'build' and config['placement_mode'] == 'full_mask':
            # A missing grid is an export defect, not an unrepresentable label.
            from build_placement import mask_for
            mask = mask_for(config, encoded, target['selection'][0], target['template'], target['angle'])
            if not mask[target['y'], target['x']]:
                raise Unrepresentable('Quantized build placement outside observed legality mask')
        return target

    def _spatial_features(self, encoded, selected):
        xy = encoded['numeric'][selected, :2].mean(axis=0)
        delta = encoded['map_coords']-xy
        return np.column_stack([delta, (delta*delta).sum(axis=1),
                                np.full(len(delta), xy[0]), np.full(len(delta), xy[1])])

    def _encode_modalities(self, encoded):
        p, d, e = self.parameters, self.config['entity_hidden'], self.config['embedding']
        features = np.concatenate([encoded['numeric'], encoded['extra_numeric']], axis=1)
        offset = features.shape[1]
        slots = encoded['entity_upgrades']
        filled = Tensor((slots.reshape(-1) > 0).astype(float)[:, None])
        upgrades = (p['upgrade_embedding'][slots.reshape(-1)]*filled).reshape((len(slots), slots.shape[1], e)).sum(axis=1)
        base = (Tensor(features) @ p['entity_w'][:offset]
                + self.type_embedding(encoded['types']) @ p['entity_w'][offset:offset+e]
                + self.type_embedding(encoded['queued_types']) @ p['entity_w'][offset+e:offset+2*e]
                + upgrades @ p['entity_w'][offset+2*e:offset+3*e]
                + p['entity_b']).tanh()
        from unit_tasks import RELATIONS
        addition = Tensor(encoded['task_numeric']) @ p['task_w']
        for i, role in enumerate(RELATIONS):
            linked = base[encoded['task_relations'][:, i]]
            addition = addition + (linked @ p['task_'+role+'_w']) * Tensor(encoded['task_relation_valid'][:, i:i+1])
            addition = addition + Tensor(encoded['task_relation_features'][:, i]) @ p['task_'+role+'_features_w']
        entity = (base+addition).tanh()
        q, k, v = (entity @ p[name] for name in ('attn_q', 'attn_k', 'attn_v'))
        count, heads = len(entity.data), self.config['attention_heads']
        positions = encoded['numeric'][:, :2]
        delta = positions[:, None, :] - positions[None, :, :]
        owners = np.asarray([e['owner'] for e in encoded['entities']] or [-1])
        relative = np.stack([delta[:, :, 0], delta[:, :, 1], (delta*delta).sum(axis=-1),
                             (owners[:, None] == owners[None, :]).astype(float)], axis=-1).reshape(count*count, 4)
        bias = Tensor(relative) @ p['relation_bias']
        attended = []
        width = d//heads
        for head in range(heads):
            sl = slice(head*width, (head+1)*width)
            scores = q[:, sl] @ k[:, sl].transpose() / math.sqrt(width)
            weights = (scores + bias[:, head].reshape((count, count))).softmax(axis=1)
            attended.append(weights @ v[:, sl])
        # concatenate() is vector-oriented; use a reshape around the head join.
        joined = concatenate([a.transpose().reshape((-1,)) for a in attended]).reshape((d, count)).transpose()
        entity = (entity + joined @ p['attn_out']).layer_norm()
        entity = (entity + (entity @ p['entity_ff']).tanh()).layer_norm()
        if encoded['entities']:
            spatial_entities = scatter_mean(entity @ p['scatter_w'], encoded['cells'], self.config['map_side']**2)
        else:
            spatial_entities = Tensor(np.zeros((self.config['map_side']**2, 8)))
        # Matrix join, via projections, keeps the existing autodiff API small.
        spatial = (Tensor(encoded['map_inputs']) @ p['map_input_w'][:MAP_CHANNELS]
                   + spatial_entities @ p['map_input_w'][MAP_CHANNELS:] + p['map_input_b']).tanh()
        spatial = (spatial + (conv2d_same(spatial, p['map_conv'], self.config['map_side'])
                              + p['map_conv_b']).tanh()).layer_norm()
        scalar = Tensor(encoded['scalars']) @ p['scalar_w'] + p['scalar_b']
        for i, prefix in enumerate(('task_command', 'task_validation')):
            scalar = scalar + Tensor(encoded['task_commands'][i]) @ p[prefix+'_w']
            embedding = self.type_embedding(encoded['task_command_templates'][i:i+1])[0]
            scalar = scalar + (embedding @ p[prefix+'_template_w']) * float(encoded['task_command_template_present'][i])
        faction, general = encoded['player_identity']
        scalar = scalar + (p['base_faction_embedding'][faction]+p['general_embedding'][general]) @ p['player_identity_w']
        scalar = scalar + Tensor(encoded['player_flags']) @ p['player_flags_w']
        return scalar.tanh(), entity, spatial

    def _trunk(self, encoded):
        scalar, entity, spatial = self._encode_modalities(encoded)
        p = self.parameters
        query = scalar @ p['pool_query']
        def pool(mask):
            indices = np.flatnonzero(mask)
            if not len(indices):
                return Tensor(np.zeros(self.config['entity_hidden']))
            selected = entity[indices]
            return (selected @ query).softmax() @ selected
        entities = encoded['entities']
        enemy = np.asarray([not e['own'] and e['owner'] >= 2 for e in entities] or [False])
        visible = np.asarray([e['visible'] for e in entities] or [False])
        neutral = np.asarray([not e['own'] and e['owner'] < 2 for e in entities] or [False])
        map_pool = (spatial @ (scalar @ p['map_pool_query'])).softmax() @ spatial
        features = concatenate([scalar, pool(encoded['own']), pool(enemy & visible), pool(enemy & ~visible),
                                pool(neutral), map_pool])
        hidden = ((features @ p['event_input_w'] + p['event_input_b']).tanh()
                  @ p['event_hidden_w'] + p['event_hidden_b']).tanh()
        value = (((hidden @ p['value_hidden_w'] + p['value_hidden_b']).tanh()) @ p['value_w'] + p['value_b'])[0]
        return hidden, entity, spatial, value

    def decide(self, encoded, memory=None, *, teacher=None, rng=None, deterministic=False):
        """One decision: DEFER plus delay, or one command with arguments plus delay."""
        from event_schedule import delay_mask
        hidden, entity, spatial, value = self._trunk(encoded)
        p, config, trace = self.parameters, self.config, {}
        logp, entropy = Tensor(0), Tensor(0)
        def draw(name, logits, mask, override=None):
            nonlocal logp, entropy
            choice = override if override is not None else (teacher.get(name) if teacher is not None else None)
            choice, lp, ent = categorical(logits, mask, choice=choice, rng=rng, deterministic=deterministic)
            logp, entropy = logp+lp, entropy+ent
            trace[name] = choice
            return choice
        delays = delay_mask(config, encoded.get('event_rate', 60))
        first_delay = int(np.flatnonzero(delays)[0])
        verb = draw('decision', hidden @ p['verb_w']+p['verb_b'], self.verb_mask(encoded),
                    KINDS.index(teacher['kind']) if teacher is not None else None)
        if verb == 0:
            context = (hidden @ p['context_w']+p['verb_embedding'][0]).tanh()
            delay = draw('delay', context @ p['delay_w']+p['delay_b'], delays,
                         teacher.get('delay', first_delay) if teacher else None)
            return [], {'kind': 'wait', 'delay': delay}, hidden, value, logp, entropy
        kind = KINDS[verb]
        context = (hidden @ p['context_w']+p['verb_embedding'][verb]).tanh()
        if kind == 'science':
            # A player-level purchase: no actor selection.
            embeddings = p['science_embedding']
            index = draw('science', embeddings @ (context @ p['science_query_w']) + p['science_b'], self.science_mask(encoded))
            context = (context+embeddings[index] @ p['science_context_w']).tanh()
            trace = {'kind': kind, 'selection': [], 'science': index}
            draw('delay', context @ p['delay_w']+p['delay_b'], delays,
                 override=teacher.get('delay', first_delay) if teacher is not None else None)
            return [{'kind': kind, 'science': config['sciences'][index]}], trace, hidden, value, logp, entropy
        actors, selected = self.actor_mask(encoded, kind), []
        limit = selection_limit(config, kind)
        selection = teacher['selection'] if teacher else None
        for ordinal in range(limit):
            scores = entity @ (context @ p['actor_pointer_w']) / math.sqrt(config['entity_hidden'])
            scores = concatenate([scores, context @ p['stop_w']+p['stop_b']])
            available = self.selection_mask(encoded, kind, selected, actors)
            actor_mask = np.concatenate([available, [ordinal > 0]])
            actor_mask[selected] = False
            forced = (selection[ordinal] if ordinal < len(selection) else len(actors)) if teacher else None
            index = draw('unit', scores, actor_mask, forced)
            if index == len(actors):
                break
            selected.append(index)
            context = (hidden @ p['context_w']+p['verb_embedding'][verb]
                       + entity[selected].mean(axis=0) @ p['selected_w']).tanh()
            available = self.selection_mask(encoded, kind, selected, actors)
            if not any(available[i] for i in range(len(actors)) if i not in selected):
                break
        trace = {'kind': kind, 'selection': selected}
        ids = [encoded['entities'][i]['id'] for i in selected]
        action = {'kind': kind}
        if kind in GROUP_KINDS:
            action['units'] = ids
        else:
            action['builder' if kind in BUILDER_KINDS else 'producer'] = ids[0]
        if kind in COMMAND_KINDS:
            embeddings = p['command_embedding']
            command = draw('command', embeddings @ (context @ p['command_query_w']) + p['command_b'],
                           self.command_mask(encoded, kind, selected))
            action['command'] = config['commands'][command]
            context = (context+embeddings[command] @ p['command_context_w']).tanh()
        if kind == 'cancel_production':
            action['slot'] = draw('slot', context @ p['slot_w']+p['slot_b'], self.slot_mask(encoded, selected))
        mode = command_target(config, trace)
        if mode == 'entity':
            index = draw('target', entity @ (context @ p['target_pointer_w'])/math.sqrt(config['entity_hidden']),
                         self.target_mask(encoded, kind, selected, trace.get('command')))
            action['target'] = encoded['entities'][index]['id']
        if kind in ('build', 'train'):
            actor = encoded['entities'][selected[0]]
            menu = config['menus'][actor['type']+'|'+kind]
            available = actor['buildable_templates' if kind == 'build' else 'trainable_templates']
            choices = [t in menu and t in available for t in config['templates']]
            embeddings = self.object_embedding(self.encoder.identity.template_objects)
            index = draw('template', embeddings @ (context @ p['template_query_w']) + p['template_b'], choices)
            action['template'] = config['templates'][index]
            context = (context+embeddings[index] @ p['template_context_w']).tanh()
        build_mask = None
        placement_unavailable = False
        if kind == 'build' and config['placement_mode'] == 'full_mask':
            from build_placement import mask_for
            angle = draw('angle', context @ p['angle_w']+p['angle_b'], np.ones(16, bool))
            context = (context+p['build_angle_embedding'][angle]).tanh()
            build_mask = mask_for(config, encoded, selected[0], index, angle)
            placement_unavailable = not build_mask.any()
            action['angle'] = math.tau*((angle+8*encoded['rotated'])%16)/16
            if placement_unavailable:
                trace.update(kind='wait', attempted_kind='build', placement_unavailable=True)
        if mode == 'position' and not placement_unavailable:
            bins, n = config['position_bins'], config['map_side']
            fine = bins//n
            cells = (spatial + Tensor(self._spatial_features(encoded, selected)) @ p['relative_w']).tanh()
            cell_mask, offset_mask = np.ones(n*n, dtype=bool), np.ones(fine*fine, dtype=bool)
            if build_mask is not None:
                from build_placement import categorical_masks, coarse_features
                cells = (cells + Tensor(coarse_features(build_mask, n)) @ p['build_map_w']).tanh()
                cell_mask, _ = categorical_masks(build_mask, n)
            scores = cells @ (context @ p['spatial_query'])/math.sqrt(config['map_hidden'])
            cell = draw('cell', scores, cell_mask)
            context = (context+cells[cell] @ p['cell_context_w']).tanh()
            if build_mask is not None:
                _, offset_mask = categorical_masks(build_mask, n, cell)
            offset = draw('offset', context @ p['offset_w']+p['offset_b'], offset_mask)
            x, y = (cell%n)*fine+offset%fine, (cell//n)*fine+offset//fine
            trace['x'], trace['y'] = x, y
            normalized = (np.asarray([x, y])+.5)/bins
            if encoded['rotated']:
                normalized = 1-normalized
            xy = encoded['low']+encoded['span']*normalized
            action['position'] = [float(xy[0]), float(xy[1]), 0]
        if kind == 'build' and config['placement_mode'] != 'full_mask':
            angle = draw('angle', context @ p['angle_w']+p['angle_b'], np.ones(config['angle_bins'], bool))
            action['angle'] = math.tau*angle/config['angle_bins']
            if encoded['rotated']:
                action['angle'] = (action['angle']+math.pi)%math.tau
        # The delay schedules the next decision after this command.
        draw('delay', context @ p['delay_w']+p['delay_b'], delays,
             override=teacher.get('delay', first_delay) if teacher is not None else None)
        return ([] if placement_unavailable else [action]), trace, hidden, value, logp, entropy

    def weight_hash(self):
        h = hashlib.sha256()
        for k, p in sorted(self.parameters.items()):
            h.update(k.encode()); h.update(p.data.tobytes())
        return h.hexdigest()

    def identity_hash(self):
        h = hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(',', ':')).encode())
        h.update(self.weight_hash().encode())
        return h.hexdigest()

    def save(self, path, optimizer=None, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {'config': self.config, 'weight_hash': self.weight_hash(),
                    'policy_identity_hash': self.identity_hash(), 'metadata': metadata or {},
                    'optimizer_step': optimizer.step_count if optimizer else 0,
                    'optimizer_lr': optimizer.lr if optimizer else None}
        arrays = {'metadata': np.asarray(json.dumps(document, allow_nan=False))}
        arrays.update({'p_' + k: p.data for k, p in self.parameters.items()})
        if optimizer:
            arrays.update({'m_' + k: v for k, v in optimizer.m.items()})
            arrays.update({'v_' + k: v for k, v in optimizer.v.items()})
        temporary = path.with_name(path.name + '.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, lr=None):
        with np.load(path, allow_pickle=False) as data:
            document = json.loads(str(data['metadata']))
            policy = cls(document['config'])
            optimizer = Adam(policy.parameters, lr or document.get('optimizer_lr') or 3e-4)
            optimizer.step_count = document['optimizer_step']
            for name, p in policy.parameters.items():
                value = data['p_' + name]
                if value.shape != p.data.shape or not np.isfinite(value).all():
                    raise ValueError('Invalid checkpoint weights')
                p.data[:] = value
                if 'm_' + name in data:
                    if not np.isfinite(data['m_' + name]).all() or not np.isfinite(data['v_' + name]).all():
                        raise ValueError('Nonfinite optimizer checkpoint state')
                    optimizer.m[name][:] = data['m_' + name]
                    optimizer.v[name][:] = data['v_' + name]
            if policy.weight_hash() != document['weight_hash'] or policy.identity_hash() != document['policy_identity_hash']:
                raise ValueError('Checkpoint weight/configuration hash mismatch')
            optimizer.parameter_steps = document['metadata'].get('event_optimizer_steps', {})
        return policy, optimizer, document
