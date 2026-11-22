"""AlphaGenerals event policy: encoder, factual action masks, teacher targets and NumPy inference.

The network is StarCraft Commander's (SCC, ICML 2021, supplement Fig. 4; notes/plan.md §1): a
group transformer over own, enemy, neutral and effect tokens, attention pooling, a spatial
ResNet, a residual LSTM core stepped once per decision, and independent heads conditioned on
one flat selected action. The PyTorch learner (torch_policy_v3.py) trains these same parameters
on whole episodes; this module is the test reference. generals_policy_old.py keeps the previous
memory-free policy runnable until adoption.
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

from generals_identity import IdentityHierarchy, in_production_scope, production_twin, validate_production_scope
from nn_numpy import Adam, Tensor, categorical, concatenate, conv2d, conv_transpose2d, scatter_sum, tensor

# v6 (re-export, Nov 2026): logit caps on every head (exact shift gradient), weapon/armour inputs,
# full production queue, no science action, visible-only enemies (observation v10). Older models
# play from their own code snapshot (their run's sources.zip), never through compatibility branches.
# The name is the stored data's contract (config architecture, part of the feature key): AlphaGenerals
# reads the same shards as the memory-free _old policy, so it keeps the name; checkpoints tell the two
# networks apart by their parameter names (policy_checkpoints).
ARCHITECTURE = 'generals-relational-v6-event'
OBSERVATION_VERSION = 11
KINDS = ('wait', 'move', 'attack_move', 'attack', 'stop', 'sell', 'build', 'train', 'rally', 'dock',
         'resume_construction', 'upgrade', 'ability', 'enter', 'exit', 'evacuate',
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
# v11 (15 Nov 2026) appends firing, poisoned and burning: what a player sees on a visible unit now.
ENTITY_FEATURES = 25
# Divides world units per second, so a fast ground unit is near 1.
VELOCITY_SCALE = 100
# Numeric columns read back when building model inputs.
OWN_COLUMN, VISIBLE_COLUMN, ENEMY_COLUMN, COOLDOWN_COLUMN, UNDER_ATTACK_COLUMN = 5, 6, 7, 14, 15
SUPPLY_COLUMN, FIRING_COLUMN, POISONED_COLUMN, BURNING_COLUMN = 21, 22, 23, 24
EXTRA_ENTITY_FEATURES = 14
ENTITY_UPGRADE_SLOTS = 4
# Upgrade/ability buttons per owned entity: (available, ready fraction) each. Vanilla
# China's command center has the most (its menu plus the faction shortcuts, 14).
ENTITY_COMMAND_SLOTS = 16
# A full supply dock holds a few hundred boxes.
SUPPLY_BOX_SCALE = 8
# Pre-match Overall ladder rating, the player's and the opponent's: (rating-centre)/scale, a
# rated flag (a ladder rating, or the player's own rating found by name) and an estimated flag
# (scripts/rating_estimator.py's joint estimate, for players with neither; re-export fix 7).
# Vanilla China players, 3 Nov 2026: median 1094, sd 291. Live play conditions on the 95th
# percentile (1730).
RATING_CENTER, RATING_SCALE, PLAY_RATING = 1100, 300, 1730
RATING_FEATURES = 6
# v11 appends the public superweapon timers, own then enemy: present, soonest ready (/SUPERWEAPON_SECONDS), ready now.
SCALAR_FEATURES = 22
SUPERWEAPON_SECONDS = 600
# v11 current perception (TrainingExporter: effects, audio_cues), one token group sharing the entity
# attention but never an actor or target: projectiles and hazards visible now, and special-power
# launch sounds heard in the last CUE_SECONDS. Columns: x, y, z, has position, velocity x/y/z,
# radius, own, enemy, projectile, hazard, cue, cue age.
EFFECT_FEATURES = 14
CUE_SECONDS = 5
EFFECT_RADIUS_SCALE = 50
# Width of the map summary (map_fc), AlphaStar's 256 whatever the entity width (15 Nov 2026).
MAP_EMBEDDING = 256
# since_decision and since_command stay in stored scalars but never reach the model:
# quiet rows are kept at most 120 ticks apart, so these clocks give away wait vs command.
HIDDEN_SCALARS = (9, 14)
# Height, cliff, supply, x, y; three visibility states; own, seen enemy, remembered enemy and neutral counts; alert.
STATIC_MAP_CHANNELS = 5
MAP_CHANNELS = 13
POSITION_BITS = 16
# Attack cooldown as AlphaStar encodes it: one-hot in 45 ms game steps, capped at 31.
COOLDOWN_BINS = 32
COOLDOWN_BIN_SECONDS = .045
MAX_COOLDOWN_SECONDS = 10
DERIVED_ENTITY_FEATURES = POSITION_BITS+COOLDOWN_BINS
# Weapon and armour inputs (re-export fix 4): the engine's three weapon slots and its damage types
# (DAMAGE_NUM_TYPES, both engine trees); the anti mask's low bits. See arms_inputs.
WEAPON_SLOTS, DAMAGE_TYPES, ANTI_BITS = 3, 38, 8
ARMS_FEATURES = WEAPON_SLOTS*(1+7+ANTI_BITS+DAMAGE_TYPES)+1+DAMAGE_TYPES
# A unit counts as under attack on the alert map for this long after taking damage.
UNDER_ATTACK_TICKS = 120
ATTENTION_LAYERS = 3
# The stored grids are this many times map_side per axis (128). The network reads them pooled MAP_POOL-fold,
# as SCC's 64x64 map, then halves its image MAP_STAGES times.
SCATTER_SCALE = 8
MAP_POOL = 2
MAP_STAGES = 2
# Up to this many entities on one scatter pixel add fully; more share that total.
SCATTER_VOLUME = 4
MAP_RESIDUAL_BLOCKS = 4
LOCATION_BLOCKS = 4
# Every head's logits, shifted so the best legal choice is 0, are soft-capped to
# (-cap, 0]: no choice is more than e^cap times less likely than the best, bounding any
# one decision's loss near cap+log(choices). The position map is capped as a whole
# before its cell/offset split; the other heads at their categorical draw. The
# upsampling stages before the position logits are layer-normalised so their gains
# cannot compound. Uncapped action-kind and angle logits reached NLL 107 and 377 in
# v10 (11 Nov 2026).
LOGIT_CAP = 50.0
# Each selected unit's key enters the unit query divided by this constant, as AlphaStar's
# mean over its fixed 512 entity slots does (12 Nov 2026: 16 is about the smallest entity count).
SELECTED_UNIT_DIVISOR = 16
# Live sampling temperature for every argument except the delay.
SAMPLING_TEMPERATURE = .8
# AlphaGenerals (notes/plan.md Â§1): SCC's policy network. Unit groups as SCC's own, enemy and
# neutral, plus the v11 effect and sound-cue tokens; the transformer and pooling run per group.
GROUPS = ('own', 'enemy', 'neutral', 'effects')
UNIT_GROUPS = 3
LSTM_LAYERS = 3
# The heads that read an action embedding of the selected action, each its own (SCC Fig. 4).
ARGUMENT_HEADS = ('delay', 'unit', 'target', 'position', 'slot', 'angle')
# Widths SCC does not give: AlphaStar's where a counterpart exists, otherwise ours (plan.md Â§6).
# 'core' is the pre-LSTM FC and every LSTM layer, chosen so the model has 45-49M parameters.
# Fixed constants. A config 'widths' entry shrinks them for tests; it is part of the feature key
# (not in event_store.MODEL_ONLY), so it never rides on a real dataset.
WIDTHS = {'entity': 256, 'ffn': 1024, 'heads': 2, 'head_width': 32, 'key': 32, 'pool_queries': 2,
          'scatter': 32, 'map_out': 32, 'map_embedding': 256, 'scalar': 512, 'scalar_out': 256,
          'core': 1024, 'action_embedding': 256, 'unit_query': 32}


def widths(config):
    """The network widths: WIDTHS, overridden only by a test model's config 'widths'."""
    return {**WIDTHS, **config.get('widths', {})}


def group_sources(group):
    """Source groups of one query group's attention blocks, in concat order: itself, then the others."""
    return [group]+[other for other in range(len(GROUPS)) if other != group]


def unit_groups(own, owners, real):
    """Each entity row's unit group: 0 own, 1 enemy (seen or remembered), 2 neutral; -1 for padding."""
    own, owners, real = np.asarray(own, dtype=bool), np.asarray(owners), np.asarray(real, dtype=bool)
    return np.where(~real, -1, np.where(own, 0, np.where(owners >= 2, 1, 2)))


def action_vocabulary(config):
    """SCC's selected action as one flat vocabulary: (kind index, object index or -1) per entry.

    The object is the production template for build/train and the command for upgrade/ability
    (the kinds whose stored target carries one); every other kind, wait included, is one entry."""
    entries = []
    for k, kind in enumerate(KINDS):
        if kind in ('build', 'train'):
            entries += [(k, t) for t in range(len(config['templates']))]
        elif kind in COMMAND_KINDS:
            entries += [(k, c) for c, name in enumerate(config['commands']) if config['command_info'][name][0] == kind]
        else:
            entries.append((k, -1))
    return np.asarray(entries, dtype=np.int64)


def action_object(target):
    """The object index of a teacher or decision target: template, command or -1."""
    if target['kind'] in ('build', 'train'):
        return target['template']
    return target['command'] if target['kind'] in COMMAND_KINDS else -1


def scalar_input_slices(config):
    """{input: rows of scalar_in_w}, in the scalar input's concatenation order (NeuralPolicy._scalars)."""
    from unit_tasks import COMMAND_FEATURES
    e = config['embedding']
    widths = [('scalars', SCALAR_FEATURES), ('player_identity', e), ('enemy_identity', e), ('ratings', RATING_FEATURES),
              ('opener', len(config['templates'])+1), ('player_flags', len(config['upgrades'])+2*len(config['sciences'])),
              ('unit_counts', len(config['types'])), ('available_actions', len(KINDS)),
              ('task_command', COMMAND_FEATURES), ('task_command_template', e),
              ('task_validation', COMMAND_FEATURES), ('task_validation_template', e)]
    slices, start = {}, 0
    for name, width in widths:
        slices[name] = slice(start, start+width)
        start += width
    return slices


def scalar_input_size(config):
    """Width of the scalar input concatenation (NeuralPolicy._scalars)."""
    return scalar_input_slices(config)['task_validation_template'].stop


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


def visible_scalars(scalars):
    """Stored scalars as the model sees them, for one row or a batch."""
    visible = np.array(scalars)
    visible[..., HIDDEN_SCALARS] = 0
    return visible


def visible_task_commands(commands):
    """Stored command receipts as the model sees them: their age is the same clock as since_command."""
    from unit_tasks import COMMANDS
    # unit_tasks.encode writes [present, *command one-hot, log age, ...]; that file is a hashed feature source.
    visible = np.array(commands)
    visible[..., 1+len(COMMANDS)] = 0
    return visible


def entity_command_inputs(p, ids, states, e):
    """Sum over an entity's command slots: ready buttons, or charging ones scaled by progress."""
    flat = ids.reshape(-1)
    ready = states[..., 0].reshape(-1, 1)
    charging = Tensor((1-ready)*(flat[:, None] > 0))
    slots = (p['entity_command_ready_embedding'][flat]*Tensor(ready)
             + (p['entity_command_charging_embedding'][flat]
                + p['entity_command_progress_embedding'][flat]*Tensor(states[..., 1].reshape(-1, 1)))*charging)
    return slots.reshape((len(ids), ids.shape[1], e)).sum(axis=1)


def rating_inputs(own, opponent):
    """Model rating inputs from two pre-match ratings.

    A rating is a number (rated), {'rating': value, 'estimated': bool} (replay_cleaning.player_ratings)
    or None (unknown). Each side gives (scaled value, rated flag, estimated flag)."""
    values = []
    for rating in (own, opponent):
        if rating is None:
            values += [0., 0., 0.]
            continue
        estimated = bool(rating['estimated']) if isinstance(rating, dict) else False
        value = rating['rating'] if isinstance(rating, dict) else rating
        values += [float(np.clip((value-RATING_CENTER)/RATING_SCALE, -3, 5)), float(not estimated), float(estimated)]
    return np.asarray(values)


UNRATED = rating_inputs(None, None)

# Strategy input z (scripts/openers.py, 17 Nov 2026): the counts of each production template among a player's
# first OPENER_BUILDINGS build orders, /OPENER_BUILDINGS, then a given flag; all zero = no opener. Shown only during
# the first OPENER_WINDOW_SECONDS of game time (95% of top-20% players order their sixth building by 222 s).
OPENER_BUILDINGS = 6
OPENER_WINDOW_SECONDS = 240


def opener_inputs(config, templates=None, seconds=0.):
    """Model opener input from a player's first OPENER_BUILDINGS build templates (policy template indices) or
    None, `seconds` of game time in."""
    values = np.zeros(len(config['templates'])+1)
    if templates is not None and seconds < OPENER_WINDOW_SECONDS:
        if len(templates) != OPENER_BUILDINGS:
            raise ValueError('An opener is exactly %d build orders' % OPENER_BUILDINGS)
        np.add.at(values, np.asarray(templates, dtype=np.int64), 1/OPENER_BUILDINGS)
        values[-1] = 1
    return values


def position_bits(xy):
    """(..., 2) normalized positions as 8 bits per axis."""
    cells = np.clip((np.asarray(xy, dtype=np.float64)*256).astype(np.int64), 0, 255).astype(np.uint8)
    return np.unpackbits(cells[..., None], axis=-1).reshape(*cells.shape[:-1], POSITION_BITS)


def derived_inputs(numeric):
    """Model-only entity inputs from stored columns: position bits and the cooldown one-hot."""
    numeric = np.asarray(numeric, dtype=np.float64)
    bins = np.minimum((numeric[..., COOLDOWN_COLUMN]/COOLDOWN_BIN_SECONDS+1e-6).astype(np.int64), COOLDOWN_BINS-1)
    return np.concatenate([position_bits(numeric[..., :2]), np.eye(COOLDOWN_BINS)[bins]], axis=-1)


# The factions the policy plays and learns from (13 Nov 2026: vanilla China plus the Tank and Nuke
# generals, which share its buildings and economy; "my general" is the player_identity input).
CHINA_FACTIONS = ('FactionChina', 'FactionChinaTankGeneral', 'FactionChinaNukeGeneral')


class TemplateTable:
    """One game's weapon and armour stats by per-game id (re-export fix 4).

    The exporter sends each template's stats once per player stream, in the observation where
    it first appears (`template_stats`). Every captured observation must pass through add(),
    selected for training or not. Times become seconds: the engines run at 30 or 60 logic frames.
    """
    def __init__(self, source_ticks_per_second):
        self.rate, self.weapons, self.armors = float(source_ticks_per_second), {}, {}

    def add(self, stats):
        for w in (stats or {}).get('weapons', ()):
            self.weapons[int(w['id'])] = [w['range'], w['min_range'], w['damage'], w['radius'],
                                          (w['min_delay_frames']+w['max_delay_frames'])/2/self.rate,
                                          w['clip_size'], w['clip_reload_frames']/self.rate,
                                          w['anti_mask'], w['damage_type']]
        for a in (stats or {}).get('armors', ()):
            if len(a['damage_multipliers']) != DAMAGE_TYPES:
                raise ValueError('Armour stats do not cover every damage type')
            self.armors[int(a['id'])] = list(a['damage_multipliers'])

    def arrays(self):
        """`weapon_stats` (W, 9) and `armor_stats` (A, DAMAGE_TYPES), rows by id; ids never sent stay zero."""
        weapons = np.zeros((max(self.weapons, default=-1)+1, 9))
        for i, row in self.weapons.items():
            weapons[i] = row
        armors = np.zeros((max(self.armors, default=-1)+1, DAMAGE_TYPES))
        for i, row in self.armors.items():
            armors[i] = row
        return {'weapon_stats': weapons, 'armor_stats': armors}


def arms_inputs(encoded):
    """Per-entity weapon and armour inputs (ARMS_FEATURES) from stored ids, bonuses and the game's table.

    Per weapon slot: present; log-scaled range x range bonus, minimum range, damage x damage bonus,
    radius, mean delay (s) / rate bonus, clip size, clip reload (s) / rate bonus; ANTI_BITS anti-mask
    bits; damage-type one-hot. Then armour present and its damage multiplier per damage type.
    """
    weapon_ids, armor_ids = np.asarray(encoded['weapon_ids']), np.asarray(encoded['armor_ids'])
    bonus = np.asarray(encoded['weapon_bonus'], dtype=np.float64)
    # No table attached (e.g. a fixture row): no stats; any used id then fails the check below.
    table = np.asarray(encoded.get('weapon_stats', np.zeros((0, 9))))
    armors = np.asarray(encoded.get('armor_stats', np.zeros((0, DAMAGE_TYPES))))
    # The exporter sends a template's stats before any entity uses its id.
    if weapon_ids.max(initial=-1) >= len(table) or armor_ids.max(initial=-1) >= len(armors):
        raise ValueError('Weapon/armour template id without stats in the game table')
    # A game without any template keeps the columns: one zero row stands in.
    table = table if len(table) else np.zeros((1, 9))
    armors = armors if len(armors) else np.zeros((1, DAMAGE_TYPES))
    count = len(armor_ids)
    blocks = []
    for slot in range(WEAPON_SLOTS):
        ids = weapon_ids[:, slot]
        present = ids >= 0
        stats = np.where(present[:, None], table[np.maximum(ids, 0)], 0)
        damage, reach, rate = bonus[:, 0], bonus[:, 1], np.maximum(bonus[:, 2], 1e-6)
        numbers = np.stack([np.log1p(stats[:, 0]*reach)/7, np.log1p(stats[:, 1])/7, np.log1p(stats[:, 2]*damage)/7,
                            np.log1p(stats[:, 3])/5, np.log1p(stats[:, 4]/rate)/3, np.log1p(stats[:, 5])/5,
                            np.log1p(stats[:, 6]/rate)/3], axis=1)*present[:, None]
        anti = ((stats[:, 7].astype(np.int64)[:, None] >> np.arange(ANTI_BITS)) & 1)*present[:, None]
        kind = np.eye(DAMAGE_TYPES)[stats[:, 8].astype(np.int64)]*present[:, None]
        blocks += [present[:, None].astype(float), numbers, anti, kind]
    worn = armor_ids >= 0
    multipliers = np.where(worn[:, None], np.clip(armors[np.maximum(armor_ids, 0)], 0, 4), 0)
    blocks += [worn[:, None].astype(float), multipliers]
    result = np.concatenate(blocks, axis=1)
    assert result.shape == (count, ARMS_FEATURES)
    return result


def count_channels(numeric, count):
    """The count plane of each real entity (own, seen enemy, remembered enemy, neutral)."""
    own, visible, enemy = (numeric[:count, column] > .5 for column in (OWN_COLUMN, VISIBLE_COLUMN, ENEMY_COLUMN))
    return np.where(own, 0, np.where(~enemy, 3, np.where(visible, 1, 2)))


def coarse_pixels(pixels, side, pool=MAP_POOL):
    """Flat pixels of a (side, side) grid as flat pixels of the grid pooled pool-fold per axis."""
    pixels = np.asarray(pixels)
    return pixels//side//pool*(side//pool)+pixels%side//pool


def pool_grid(values, side, pool=MAP_POOL):
    """(..., side*side) averaged over pool x pool blocks: (..., (side//pool)**2)."""
    values = np.asarray(values)
    coarse = side//pool
    return values.reshape(*values.shape[:-1], coarse, pool, coarse, pool).mean(axis=(-3, -1)).reshape(*values.shape[:-1], coarse*coarse)


def map_planes(encoded, pool=1):
    """(MAP_CHANNELS, side, side): terrain, visibility, entity counts and the under-attack alert.

    pool > 1: on a grid pooled pool-fold per axis (terrain and visibility shares averaged, entities recounted)."""
    numeric, pixels = np.asarray(encoded['numeric']), np.asarray(encoded['cells'])
    count = len(encoded['entities']) if 'entities' in encoded else encoded['entity_count']
    visibility = np.asarray(encoded['map_visibility'])
    full = int(math.isqrt(visibility.size))
    side = full//pool
    pixels = coarse_pixels(pixels, full, pool)
    planes = np.zeros((MAP_CHANNELS, side*side))
    planes[:STATIC_MAP_CHANNELS] = pool_grid(np.asarray(encoded['map_static']).T, full, pool)
    for category in range(3):
        planes[STATIC_MAP_CHANNELS+category] = pool_grid(visibility == category, full, pool)
    if count:
        counts = np.zeros((4, side*side))
        np.add.at(counts, (count_channels(numeric, count), pixels[:count]), 1)
        planes[STATIC_MAP_CHANNELS+3:STATIC_MAP_CHANNELS+7] = np.log1p(counts)/4
        attacked = pixels[:count][numeric[:count, UNDER_ATTACK_COLUMN] > .5]
        planes[STATIC_MAP_CHANNELS+7, attacked] = 1
    return planes.reshape(MAP_CHANNELS, side, side)


def scatter_weights(pixels, real, side):
    """Each real entity's share of its pixel: 1 up to SCATTER_VOLUME entities there."""
    counts = np.bincount(np.asarray(pixels)[real], minlength=side*side)
    return SCATTER_VOLUME/np.maximum(counts[pixels], SCATTER_VOLUME)*real


def location_order(config):
    """Flat 256x256 indices laid out as (map cell, offset within it)."""
    n, bins = config['map_side'], config['position_bins']
    fine = bins//n
    cell, offset = np.meshgrid(np.arange(n*n), np.arange(fine*fine), indexing='ij')
    return ((cell//n)*fine+offset//fine)*bins+(cell%n)*fine+offset%fine


def cap_logits(logits, mask):
    """LOGIT_CAP's soft cap on one head's logits (a Tensor), shifted by the best legal one.

    The shift is that logit itself, not a detached copy: under the cap the shift changes the
    distribution, so the gradient must flow through it to be exact."""
    legal = np.flatnonzero(np.asarray(mask, dtype=bool))
    top = logits[int(legal[np.argmax(logits.data[legal])])]
    return ((logits-top)/LOGIT_CAP).tanh()*LOGIT_CAP


def location_stages(config):
    """Output channels of each 2x upsampling from map_side to position_bins."""
    fine, channels = config['position_bins']//config['map_side'], config['map_hidden']
    stages = int(math.log2(fine))
    return [max(1, channels >> (i+1)) for i in range(stages-1)]+[1]


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
        # Special powers (v11 sound cues); 0 is an unknown power.
        self.power_ids = {name: i+1 for i, name in enumerate(config['powers'])}
        # Row 0 is the empty command slot.
        self.command_ids = {name: i+1 for i, name in enumerate(config['commands'])}
        # Per-game inputs, found by the identity of the game's terrain record.
        self._extent, self._static, self._player, self._home = None, {}, {}, None
        self._menus = {}

    def _bounds(self, terrain):
        extent = self._extent
        if extent is None or extent[0] is not terrain:
            if terrain['grid_size'] != self.config['map_side']*SCATTER_SCALE:
                raise ValueError('Observation grid differs from the policy map')
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
            return entry[1]
        side = terrain['grid_size']
        height, cliff = np.asarray(terrain['height'], dtype=float), np.asarray(terrain['cliff'], dtype=float)
        sources = terrain.get('supply_sources', [])
        if rotated:
            height, cliff = (grid.reshape(side, side)[::-1, ::-1].reshape(-1) for grid in (height, cliff))
            sources = [[*(low+high-np.asarray(s['position'][:2])).tolist(), *s['position'][2:]] for s in sources]
        else:
            sources = [s['position'] for s in sources]
        base = np.zeros((side*side, STATIC_MAP_CHANNELS))
        base[:, 0] = height / 512
        base[:, 1] = cliff
        # Public static supply positions are available before scouting. Never
        # import supply quantities or enemy entities from the global simulator.
        for source in sources:
            pos = (np.asarray(source[:2])-low)/span
            x, y = np.clip((pos*side).astype(int), 0, side-1)
            base[y*side+x, 2] += 1
        base[:, 2] = np.log1p(base[:, 2]) / 4
        base[:, 3:5] = (np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2) + .5) / side
        self._static[rotated] = (terrain, base)
        return base

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

    @staticmethod
    def _superweapons(observation):
        """Public timers (every player's, as the UI shows them), own then enemy: present, soonest, ready."""
        me, values = observation['player']['index'], []
        for own in (True, False):
            timers = [t for t in observation['superweapons'] if (t['owner'] == me) == own]
            values += [float(bool(timers)),
                       min(t['ready_seconds'] for t in timers)/SUPERWEAPON_SECONDS if timers else 0.,
                       float(any(t['ready'] for t in timers))]
        return np.minimum(np.asarray(values), 1.)

    def _effects(self, observation, low, high, span, rotated, source_ticks_per_second):
        """(numeric (n, EFFECT_FEATURES), catalog types, power ids): v11's projectiles, hazards and sound cues.

        Ordered without ids, like entities; positions and velocities in the canonical (mirrored) frame."""
        me = observation['player']['index']
        mirror = -1. if rotated else 1.
        rows, types, powers = [], [], []
        def place(position):
            if position is None:
                return [0., 0., 0., 0.]
            xy = (low+high-np.asarray(position[:2], dtype=float)) if rotated else np.asarray(position[:2], dtype=float)
            xy = (xy-low)/span
            return [float(xy[0]), float(xy[1]), position[2]/512, 1.]
        effects = sorted(observation['effects'], key=lambda e: (e['kind'], e['type'], tuple(e['position']), e['owner']))
        for e in effects:
            vx, vy, vz = (v*source_ticks_per_second/VELOCITY_SCALE for v in e['velocity'])
            rows.append(place(e['position']) + [mirror*vx, mirror*vy, vz, min(e['radius']/EFFECT_RADIUS_SCALE, 1.),
                        float(e['owner'] == me), float(e['owner'] >= 2 and e['owner'] != me),
                        float(e['kind'] == 'projectile'), float(e['kind'] == 'hazard'), 0., 0.])
            types.append(self.types.get(e['type'], 0)); powers.append(0)
        cues = sorted(observation['audio_cues'], key=lambda c: (c['power'], c['owner'], c['ticks_since']))
        for c in cues:
            age = min(c['ticks_since']/source_ticks_per_second/CUE_SECONDS, 1.)
            rows.append(place(c['position']) + [0., 0., 0., 0., float(c['owner'] == me),
                        float(c['owner'] >= 2 and c['owner'] != me), 0., 0., 1., age])
            types.append(0); powers.append(self.power_ids.get(c['power'], 0))
        return (np.asarray(rows, dtype=float).reshape(-1, EFFECT_FEATURES), np.asarray(types, dtype=int),
                np.asarray(powers, dtype=int))

    def encode(self, observation, terrain, frame, *, previous_frame=None, home_position=None,
               source_ticks_per_second=60):
        if observation.get('observation_policy_version') != OBSERVATION_VERSION:
            raise ValueError('Policy requires observation version %d' % OBSERVATION_VERSION)
        if observation['player']['faction'] not in CHINA_FACTIONS:
            raise ValueError('Policy is for vanilla, Tank and Nuke China')
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
        side = terrain['grid_size']
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
                           'visibility_grid': np.asarray(observation['visibility_grid']).reshape(side, side)[::-1, ::-1].reshape(-1)}
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
            since_damage = column('ticks_since_damage', -1)
            mirror, facing = -1. if rotated else 1., column('facing', 0)
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
                np.asarray([min(sum(item['remaining_quantity'] for item in q), 20) / 20 for q in queues], dtype=float),
                # Exported for owned entities only. Native frames become seconds and canonical ticks.
                # A weapon that cannot fire again reports a huge next-shot frame.
                np.minimum(column('weapon_cooldown_ticks', 0) / source_ticks_per_second, MAX_COOLDOWN_SECONDS),
                np.logical_and(since_damage >= 0, since_damage*60/source_ticks_per_second <= UNDER_ATTACK_TICKS).astype(float),
                # Mirrored maps reverse directions. A hidden sighting keeps its last facing and rank, but its motion is unknown.
                mirror*column('velocity_x', 0)*source_ticks_per_second/VELOCITY_SCALE*np.asarray(visibles, dtype=float),
                mirror*column('velocity_y', 0)*source_ticks_per_second/VELOCITY_SCALE*np.asarray(visibles, dtype=float),
                mirror*np.sin(facing), mirror*np.cos(facing),
                column('veterancy', 0) / 3,
                # Public: a seen pile's depletion; a fogged sighting keeps its last count.
                np.log1p(column('supply_boxes')) / SUPPLY_BOX_SCALE,
                # v11: visible now (the exporter clears them on fog ghosts).
                column('firing'), column('poisoned'), column('burning')])
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
            [np.log1p(since_command)/5, source_ticks_per_second/60],
            self._superweapons(observation)])
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
        visibility = np.asarray(observation['visibility_grid'], dtype=np.uint8)
        base = self._static_map(terrain, low, high, span, rotated)
        upgrade_ids, catalog_types = self.upgrade_ids, self.types
        entity_upgrades = np.zeros((max(real_count, 1), ENTITY_UPGRADE_SLOTS), dtype=int)
        entity_commands = np.zeros((max(real_count, 1), ENTITY_COMMAND_SLOTS), dtype=int)
        command_states = np.zeros((max(real_count, 1), ENTITY_COMMAND_SLOTS, 2))
        command_ids = self.command_ids
        info = self.config['command_info']
        grid_xy = np.clip((numeric[:real_count, :2] * side).astype(int), 0, side-1)
        cells = (grid_xy[:, 1]*side+grid_xy[:, 0]).tolist()
        # Every queue slot (re-export fix 6): a unit type (catalog id, -1 when the slot holds none)
        # or an upgrade (upgrade id, 0 when none); slot 0 is the item in production now.
        queue_units = np.full((max(real_count, 1), QUEUE_SLOTS), -1, dtype=int)
        queue_upgrades = np.zeros((max(real_count, 1), QUEUE_SLOTS), dtype=int)
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
            # Each button's own state: ready now, or charging with its progress
            # (Red Guard capture, Dragon Tank firewall, general's powers).
            # Evacuate is outside the command vocabulary.
            slots = sorted([(command_ids[name], 1., 1.) for name in commands if name in command_ids]
                           + [(command_ids[c['command']], 0., min(max(c['ready_percent'], 0), 100)/100)
                              for c in charging if c['command'] in command_ids])
            if len(slots) > ENTITY_COMMAND_SLOTS:
                raise ValueError('Entity has more commands than command slots')
            if slots:
                entity_commands[index, :len(slots)] = [slot[0] for slot in slots]
                command_states[index, :len(slots)] = [slot[1:] for slot in slots]
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
            for slot, item in enumerate(e.get('production_queue', [])[:QUEUE_SLOTS]):
                if item.get('unit_type'):
                    queue_units[index, slot] = catalog_types.get(item['unit_type'], 0)
                if item.get('upgrade'):
                    queue_upgrades[index, slot] = upgrade_ids.get(item['upgrade'], 0)
        if real_count:
            extra = np.zeros((real_count, EXTRA_ENTITY_FEATURES))
            if extra_rows:
                extra[extra_index] = extra_rows
        else:
            extra = np.asarray([[0]*EXTRA_ENTITY_FEATURES])
        # Weapon/armour template ids into the game's TemplateTable, and an owned unit's fire bonus.
        weapon_ids = np.full((max(real_count, 1), WEAPON_SLOTS), -1, dtype=int)
        armor_ids = np.full(max(real_count, 1), -1, dtype=int)
        weapon_bonus = np.ones((max(real_count, 1), 3))
        for index, e in enumerate(entities):
            weapons = e.get('weapons')
            if weapons:
                weapon_ids[index] = weapons[:WEAPON_SLOTS]
            armor_ids[index] = e.get('armor', -1)
            if e['own'] and e.get('weapon_bonus'):
                weapon_bonus[index] = e['weapon_bonus']
        player_flags, purchasable_sciences = self._player_flags(p)
        encoded.update(weapon_ids=weapon_ids, armor_ids=armor_ids, weapon_bonus=weapon_bonus,
                       # Empty until the game's table is attached (shards: episode shared data; live: the actor).
                       **TemplateTable(1).arrays())
        effect_numeric, effect_types, effect_powers = self._effects(observation, low, high, span, rotated,
                                                                    source_ticks_per_second)
        encoded.update(effect_numeric=effect_numeric, effect_types=effect_types, effect_powers=effect_powers)
        encoded.update(map_visibility=visibility, map_static=base, cells=np.asarray(cells or [0]),
                       extra_numeric=extra,
                       queue_units=queue_units, queue_upgrades=queue_upgrades,
                       entity_upgrades=entity_upgrades, entity_commands=entity_commands,
                       entity_command_states=command_states, player_flags=player_flags,
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
        if n <= 0 or 32 % n or bins % n or bins <= n or (bins//n) & (bins//n-1):
            raise ValueError('Invalid map/attention dimensions')
        if config['delay_ticks'] != list(range(1, config['event_max_delay']+1)):
            raise ValueError('Unsupported delay vocabulary')
        if not np.isfinite(config.get('variant_l2', 0)) or config.get('variant_l2', 0) < 0:
            raise ValueError('Variant regularization must be finite and nonnegative')
        if config['placement_mode'] == 'full_mask' and config['angle_bins'] != 16:
            raise ValueError('Build angles must match the native 16-bin vocabulary')
        # SCC's selected action: one flat vocabulary of (kind, object) entries (action_vocabulary).
        self.actions = action_vocabulary(config)
        self.action_index = {(int(k), int(o)): i for i, (k, o) in enumerate(self.actions)}
        self._template_positions = {name: i for i, name in enumerate(config['templates'])}
        if not parameters:
            return
        rng = np.random.default_rng(seed)
        identity_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        W = widths(config)
        d, e, m, n = W['entity'], config['embedding'], config['map_hidden'], config['map_side']
        inner, key, core, a = W['heads']*W['head_width'], W['key'], W['core'], W['action_embedding']
        def w(name, shape, gain=1, *, generator=None):
            self.parameters[name] = Tensor((generator or rng).normal(0, gain/math.sqrt(shape[0]), shape), requires_grad=True)
        def b(name, count, value=0):
            self.parameters[name] = Tensor(np.full(count, value), requires_grad=True)
        def c(name, outputs, inputs, size, gain=math.sqrt(2)):
            self.parameters[name] = Tensor(rng.normal(0, gain/math.sqrt(inputs*size*size), (outputs, inputs, size, size)),
                                           requires_grad=True)
            b(name+'_b', (outputs, 1, 1))
        def norm(name, size):
            b(name+'_g', size, 1.); b(name+'_b', size)
        vocab = self.encoder.identity.config
        # Unit input (plan Â§1a): type embedding, position and every other entity feature.
        # Embedding scale depends on vector width, not vocabulary size.
        self.parameters['family_embedding'] = Tensor(
            identity_rng.normal(0, 1/math.sqrt(e), (len(vocab['families']), e)), requires_grad=True)
        self.parameters['base_faction_embedding'] = Tensor(
            identity_rng.normal(0, .25/math.sqrt(e), (len(vocab['base_factions']), e)), requires_grad=True)
        self.parameters['base_faction_embedding'].data[0] = 0
        b('general_embedding', (len(vocab['generals']), e))
        b('variant_embedding', (len(vocab['objects']), e))
        w('entity_w', (ENTITY_FEATURES+EXTRA_ENTITY_FEATURES+DERIVED_ENTITY_FEATURES+ARMS_FEATURES+5*e, d)); b('entity_b', d)
        # Row 0 is the empty upgrade slot; it is masked out of every sum.
        w('upgrade_embedding', (len(config['upgrades'])+1, e), .5)
        # v11 effect and sound-cue tokens (Encoder._effects); power 0 is none or unknown.
        w('effect_w', (EFFECT_FEATURES, d)); b('effect_b', d)
        w('effect_type_w', (e, d), .5); w('effect_power_w', (e, d), .5)
        w('power_embedding', (len(config['powers'])+1, e), .5)
        self.parameters['power_embedding'].data[0] = 0
        self.parameters['upgrade_embedding'].data[0] = 0
        # Per-entity buttons: ready, or charging plus progress. Row 0 is the empty slot.
        for name in ('entity_command_ready', 'entity_command_charging', 'entity_command_progress'):
            w(name+'_embedding', (len(config['commands'])+1, e), .5)
            self.parameters[name+'_embedding'].data[0] = 0
        from unit_tasks import NUMERIC, RELATIONS, RELATION_FEATURES, COMMAND_FEATURES, VALIDATION_COLUMNS
        w('task_w', (NUMERIC, d), .5)
        self.parameters['task_w'].data[list(VALIDATION_COLUMNS)] = 0
        for role in RELATIONS:
            w('task_'+role+'_w', (d, d), .25)
            w('task_'+role+'_features_w', (RELATION_FEATURES, d), .25)
        self.parameters['task_validation_target_w'].data[:] = 0
        self.parameters['task_validation_target_features_w'].data[:] = 0
        # Group transformer (plan Â§1b): per layer and query group, one self- and three cross-attention
        # blocks with their own weights, concatenated into the group's FC; pre-norm, per-group FFN.
        for layer in range(ATTENTION_LAYERS):
            for g, group in enumerate(GROUPS):
                name = f'attn{layer}_{group}'
                norm(name+'_ln', d)
                for s in group_sources(g):
                    for projection in ('q', 'k', 'v'):
                        w(f'{name}_{GROUPS[s]}_{projection}', (d, inner))
                w(name+'_out_w', (len(GROUPS)*inner, d), .5); b(name+'_out_b', d)
                norm(f'ffn{layer}_{group}_ln', d)
                w(f'ffn{layer}_{group}_in_w', (d, W['ffn']), math.sqrt(2)); b(f'ffn{layer}_{group}_in_b', W['ffn'])
                w(f'ffn{layer}_{group}_out_w', (W['ffn'], d), .5); b(f'ffn{layer}_{group}_out_b', d)
        # Group outputs (plan Â§1c): final norm, attention pooling, attention keys and the unit maps.
        for g, group in enumerate(GROUPS):
            norm(f'final_{group}_ln', d)
            self.parameters[f'pool_{group}_query'] = Tensor(rng.normal(0, 1, (W['pool_queries'], inner)), requires_grad=True)
            w(f'pool_{group}_k', (d, inner)); w(f'pool_{group}_v', (d, inner))
            if g < UNIT_GROUPS:
                w(f'key_{group}_w', (d, key)); b(f'key_{group}_b', key)
                w(f'scatter_{group}_w', (d, W['scatter']), .5); b(f'scatter_{group}_b', W['scatter'])
        # Spatial (plan Â§1d): map planes and unit maps, our ResNet, a 1x1 conv, flatten and FC.
        scatter = W['scatter']
        c('map_project', scatter, MAP_CHANNELS+scatter, 1)
        c('map_down0', max(1, m//2), scatter, 4)
        c('map_down1', m, max(1, m//2), 4)
        for block in range(MAP_RESIDUAL_BLOCKS):
            c(f'map_res{block}_a', m, m, 3)
            c(f'map_res{block}_b', m, m, 3)
        c('map_out', W['map_out'], m, 1)
        w('map_fc_w', (n*n*W['map_out'], W['map_embedding']), math.sqrt(2)); b('map_fc_b', W['map_embedding'])
        # Scalars (plan Â§1e): every scalar input concatenated, FC-ReLU, FC-ReLU. The task validation
        # receipt starts with no effect, as its entity counterpart (task_w's validation columns).
        w('scalar_in_w', (scalar_input_size(config), W['scalar']), math.sqrt(2)); b('scalar_in_b', W['scalar'])
        rows = scalar_input_slices(config)
        self.parameters['scalar_in_w'].data[rows['task_validation'].start:rows['task_validation_template'].stop] = 0
        w('scalar_hidden_w', (W['scalar'], W['scalar_out']), math.sqrt(2)); b('scalar_hidden_b', W['scalar_out'])
        # Core: FC-ReLU, then a residual LSTM of LSTM_LAYERS layer-normed layers (AlphaStar's internals).
        pooled = len(GROUPS)*W['pool_queries']*inner
        w('core_in_w', (W['scalar_out']+pooled+W['map_embedding'], core), math.sqrt(2)); b('core_in_b', core)
        for layer in range(LSTM_LAYERS):
            w(f'core_lstm{layer}_x_w', (core, 4*core)); w(f'core_lstm{layer}_h_w', (core, 4*core))
            # Gate order (input, forget, cell, output); each gate normalised on its own; forget bias 1.
            self.parameters[f'core_lstm{layer}_gate_g'] = Tensor(np.ones((4, core)), requires_grad=True)
            gate_bias = np.zeros((4, core))
            gate_bias[1] = 1
            self.parameters[f'core_lstm{layer}_gate_b'] = Tensor(gate_bias, requires_grad=True)
            norm(f'core_lstm{layer}_cell', core)
        # Heads (plan Â§1f): every head reads the core output; the selected action is one flat softmax,
        # and each argument head has its own embedding of that action.
        actions = len(self.actions)
        w('action_w', (core, actions), .1); b('action_b', actions)
        for head in ARGUMENT_HEADS:
            w(head+'_action_embedding', (actions, a), 1)
        delays = len(config['delay_ticks'])
        w('delay_w', (core+a, delays), .1); b('delay_b', delays)
        # Selected units: a pointer network; the query LSTM steps once per chosen unit (end token last).
        uq = W['unit_query']
        w('unit_in_w', (core+a, a)); b('unit_in_b', a)
        w('unit_ar_w', (key, a), .5); b('unit_ar_b', a)
        w('unit_query_w', (a, uq), .5); b('unit_query_b', uq)
        w('unit_lstm_input_w', (uq, 4*uq), .5); w('unit_lstm_hidden_w', (uq, 4*uq), .5); b('unit_lstm_b', 4*uq)
        w('unit_score_q_w', (uq, a)); w('unit_score_k_w', (key, a)); b('unit_score_b', a)
        self.parameters['unit_end_key'] = Tensor(rng.normal(0, 1/math.sqrt(key), key), requires_grad=True)
        summary = 2*d
        w('target_in_w', (core+a+summary, a)); b('target_in_b', a)
        w('target_score_k_w', (key, a))
        w('slot_w', (core+a+summary, QUEUE_SLOTS), .1); b('slot_b', QUEUE_SLOTS)
        w('angle_w', (core+a, config['angle_bins']), .1); b('angle_b', config['angle_bins'])
        # Target position: a query reshaped to 4 n x n planes beside the 1x1 map, FiLM blocks, upsampling.
        w('location_query_w', (core+a+summary, 4*n*n), .5); b('location_query_b', 4*n*n)
        w('location_film_w', (4*n*n, 2*LOCATION_BLOCKS*m), .1); b('location_film_b', 2*LOCATION_BLOCKS*m)
        c('location_input', m, W['map_out']+4, 1)
        for block in range(LOCATION_BLOCKS):
            c(f'location_block{block}_a', m, m, 1)
            c(f'location_block{block}_b', m, m, 3)
        if config['placement_mode'] == 'full_mask':
            w('build_map_w', (1, m), .2)
        channels = m
        stages = location_stages(config)
        for stage, outputs in enumerate(stages):
            # Transposed kernels are (in, out, K, K); the final logits start small.
            self.parameters[f'location_up{stage}'] = Tensor(rng.normal(
                0, (.1 if stage == len(stages)-1 else math.sqrt(2))/math.sqrt(channels*16), (channels, outputs, 4, 4)), requires_grad=True)
            b(f'location_up{stage}_b', (outputs, 1, 1))
            channels = outputs
        # Value: untrained in imitation (RL replaces it).
        w('value_w', (core, 1), .1); b('value_b', 1)

    def object_embedding(self, objects):
        """Shared family/base faction plus zero-initialized general/type residuals."""
        identity, p = self.encoder.identity, self.parameters
        return (p['family_embedding'][identity.families[objects]]
                + p['base_faction_embedding'][identity.base_factions[objects]]
                + p['general_embedding'][identity.generals[objects]]
                + p['variant_embedding'][objects])

    def type_embedding(self, types):
        return self.object_embedding(self.encoder.identity.type_objects[types])

    def queue_inputs(self, units, upgrades):
        """(current, pending): the item in production now (queue slot 0) and the sum of the rest,
        each a unit type embedding or an upgrade embedding; empty slots add nothing."""
        p, e = self.parameters, self.config['embedding']

        def items(u, g):
            u, g = np.asarray(u), np.asarray(g)
            unit = self.type_embedding(np.maximum(u, 0)) * Tensor((u >= 0).astype(float)[:, None])
            upgrade = p['upgrade_embedding'][g] * Tensor((g > 0).astype(float)[:, None])
            return unit + upgrade
        rows, slots = units.shape
        current = items(units[:, 0], upgrades[:, 0])
        pending = items(units[:, 1:].reshape(-1), upgrades[:, 1:].reshape(-1)).reshape((rows, slots-1, e)).sum(axis=1)
        return current, pending

    def regularization_loss(self):
        if not self.config.get('variant_l2', 0):
            return Tensor(0)
        # Apply once per optimizer update. There is no shrinkage on the shared
        # families or base faction representations.
        loss = Tensor(0)
        for name in ('general_embedding', 'variant_embedding'):
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

    def production_available(self, actor, kind):
        """The actor's current build/train options as vanilla twins (generals_identity.production_twin)."""
        return {production_twin(self.config, name)
                for name in actor['buildable_templates' if kind == 'build' else 'trainable_templates']}

    def production_native(self, actor, kind, template):
        """The actor's own template whose twin is `template`: what a Tank/Nuke player sends to the engine."""
        for name in actor['buildable_templates' if kind == 'build' else 'trainable_templates']:
            if production_twin(self.config, name) == template:
                return name
        return template

    def _production_menu(self, name, kind):
        """The templates this type offers for `kind`, within the production scope."""
        key = (name, kind)
        menu = self._menus.get(key)
        if menu is None:
            name = production_twin(self.config, name)
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
                    passes = not menu.isdisjoint(self.production_available(e, kind))
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
                template = production_twin(config, a['template'])
                target['template'] = config['templates'].index(template)
                actor = encoded['entities'][selection[0]]
                if template not in config['menus'].get(production_twin(config, actor['type']) + '|' + kind, []):
                    raise Unrepresentable('Template outside producer menu')
                if template not in self.production_available(actor, kind):
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

    def action_choice(self, target):
        """The flat action index of a teacher or decision target."""
        return self.action_index[(KINDS.index(target['kind']), action_object(target))]

    def object_masks(self, encoded):
        """{kind: objects some own unit offers} for the object kinds, from the stored arrays alone.

        A template counts when a complete own unit has it on its (production-scope) menu and has
        any choice available now; a command when a live complete own unit has its button ready.
        The stored arrays hold no per-unit prerequisites, so this is a superset of the exact
        choices; the teacher's object is always in it (test_alphagenerals)."""
        config = self.config
        numeric, extra = np.asarray(encoded['numeric']), np.asarray(encoded['extra_numeric'])
        own = np.asarray(encoded['own'], dtype=bool)
        complete = own & (numeric[:, 9] > .5)
        alive = complete & (numeric[:, 3] > 0)
        types = np.asarray(encoded['types'])
        index = self._template_positions
        masks = {}
        for kind, column in (('build', 2), ('train', 3)):
            mask = np.zeros(len(config['templates']), dtype=bool)
            for t in np.unique(types[complete & (extra[:, column] > 0)]):
                mask[[index[name] for name in self._production_menu(config['types'][t], kind) if name in index]] = True
            masks[kind] = mask
        ids, states = np.asarray(encoded['entity_commands'])[alive], np.asarray(encoded['entity_command_states'])[alive]
        ready = np.unique(ids[(states[..., 0] > .5) & (ids > 0)])-1
        info = config['command_info']
        for kind in COMMAND_KINDS:
            mask = np.zeros(len(config['commands']), dtype=bool)
            mask[[c for c in ready if info[config['commands'][c]][0] == kind]] = True
            masks[kind] = mask
        return masks

    def action_mask(self, encoded, verb=None):
        """Flat action mask: the kind mask (verb_mask, or the stored one) times object_masks."""
        verb = np.asarray(self.verb_mask(encoded) if verb is None else verb, dtype=bool)
        mask = verb[self.actions[:, 0]].copy()
        objects = self.object_masks(encoded)
        for kind, values in objects.items():
            rows = self.actions[:, 0] == KINDS.index(kind)
            mask[rows] &= values[self.actions[rows, 1]]
        return mask

    def object_actors(self, encoded, kind, obj):
        """Owned units that can perform `kind` with this object now (live play: exact)."""
        mask = self.actor_mask(encoded, kind)
        for i, e in enumerate(encoded['entities']):
            if not mask[i]:
                continue
            if kind in ('build', 'train'):
                template = self.config['templates'][obj]
                mask[i] = template in self._production_menu(e['type'], kind) and template in self.production_available(e, kind)
            else:
                mask[i] = obj in self._commands(e, kind)
        return mask

    def exact_action_mask(self, encoded):
        """action_mask narrowed to the objects some actor can use now (live play only)."""
        mask = self.action_mask(encoded)
        for i in np.flatnonzero(mask & (self.actions[:, 1] >= 0)):
            k, obj = self.actions[i]
            mask[i] = bool(self.object_actors(encoded, KINDS[k], int(obj)).any())
        return mask

    # AlphaGenerals network (notes/plan.md Â§1), one observation at a time. torch_policy_v3 batches
    # the same equations; test_alphagenerals holds them equal.

    def _layer_norm(self, x, name):
        p = self.parameters
        return x.layer_norm()*p[name+'_g']+p[name+'_b']

    def _conv(self, values, name, stride=1, padding=0):
        return conv2d(values, self.parameters[name], stride, padding)+self.parameters[name+'_b']

    def _attend(self, q, k, v):
        """Multi-head attention of query rows over source rows, heads concatenated."""
        W = widths(self.config)
        width, parts = W['head_width'], []
        for head in range(W['heads']):
            sl = slice(head*width, (head+1)*width)
            weights = (q[:, sl] @ k[:, sl].transpose()/math.sqrt(width)).softmax(axis=1)
            parts.append(weights @ v[:, sl])
        return concatenate(parts, axis=1)

    def _entity_inputs(self, encoded):
        """(units (count, d), effect tokens (k, d) or None): each unit's embedded input (plan Â§1a)."""
        p, e = self.parameters, self.config['embedding']
        features = np.concatenate([encoded['numeric'], encoded['extra_numeric'],
                                   derived_inputs(encoded['numeric']), arms_inputs(encoded)], axis=1)
        offset = features.shape[1]
        slots = encoded['entity_upgrades']
        filled = Tensor((slots.reshape(-1) > 0).astype(float)[:, None])
        upgrades = (p['upgrade_embedding'][slots.reshape(-1)]*filled).reshape((len(slots), slots.shape[1], e)).sum(axis=1)
        commands = entity_command_inputs(p, encoded['entity_commands'], encoded['entity_command_states'], e)
        current, pending = self.queue_inputs(encoded['queue_units'], encoded['queue_upgrades'])
        base = (Tensor(features) @ p['entity_w'][:offset]
                + self.type_embedding(encoded['types']) @ p['entity_w'][offset:offset+e]
                + current @ p['entity_w'][offset+e:offset+2*e]
                + pending @ p['entity_w'][offset+2*e:offset+3*e]
                + upgrades @ p['entity_w'][offset+3*e:offset+4*e]
                + commands @ p['entity_w'][offset+4*e:offset+5*e]
                + p['entity_b']).tanh()
        from unit_tasks import RELATIONS
        addition = Tensor(encoded['task_numeric']) @ p['task_w']
        for i, role in enumerate(RELATIONS):
            linked = base[encoded['task_relations'][:, i]]
            addition = addition + (linked @ p['task_'+role+'_w']) * Tensor(encoded['task_relation_valid'][:, i:i+1])
            addition = addition + Tensor(encoded['task_relation_features'][:, i]) @ p['task_'+role+'_features_w']
        units = (base+addition).tanh()
        effects = None
        if len(encoded['effect_numeric']):
            effects = (Tensor(encoded['effect_numeric']) @ p['effect_w']
                       + self.type_embedding(encoded['effect_types']) @ p['effect_type_w']
                       + p['power_embedding'][encoded['effect_powers']] @ p['effect_power_w'] + p['effect_b']).tanh()
        return units, effects

    def _group_transformer(self, rows):
        """Per-group features (Tensor (n_g, d), or None for an empty group) after the plan Â§1b layers."""
        p, W = self.parameters, widths(self.config)
        inner = W['heads']*W['head_width']
        x = list(rows)
        for layer in range(ATTENTION_LAYERS):
            normed = [None if value is None else self._layer_norm(value, f'attn{layer}_{GROUPS[g]}_ln')
                      for g, value in enumerate(x)]
            updated = []
            for g, value in enumerate(x):
                if value is None:
                    updated.append(None)
                    continue
                name, ffn = f'attn{layer}_{GROUPS[g]}', f'ffn{layer}_{GROUPS[g]}'
                parts = []
                for s in group_sources(g):
                    if normed[s] is None:  # An empty source group contributes zeros.
                        parts.append(Tensor(np.zeros((len(value.data), inner))))
                        continue
                    pair = f'{name}_{GROUPS[s]}'
                    parts.append(self._attend(normed[g] @ p[pair+'_q'], normed[s] @ p[pair+'_k'], normed[s] @ p[pair+'_v']))
                value = value+concatenate(parts, axis=1) @ p[name+'_out_w']+p[name+'_out_b']
                hidden = (self._layer_norm(value, ffn+'_ln') @ p[ffn+'_in_w']+p[ffn+'_in_b']).relu()
                updated.append(value+hidden @ p[ffn+'_out_w']+p[ffn+'_out_b'])
            x = updated
        return [None if value is None else self._layer_norm(value, f'final_{GROUPS[g]}_ln') for g, value in enumerate(x)]

    def _pool(self, value, group):
        """Attention pooling (SCC Â§A.3): learned queries over a group's units, heads flattened."""
        p, W = self.parameters, widths(self.config)
        heads, width, queries = W['heads'], W['head_width'], W['pool_queries']
        if value is None:
            return Tensor(np.zeros(queries*heads*width))
        k, v, q = value @ p[f'pool_{group}_k'], value @ p[f'pool_{group}_v'], p[f'pool_{group}_query']
        parts = []
        for j in range(queries):
            for head in range(heads):
                sl = slice(head*width, (head+1)*width)
                parts.append((k[:, sl] @ q[j, sl]/math.sqrt(width)).softmax() @ v[:, sl])
        return concatenate(parts)

    @staticmethod
    def _entity_order(values, members, count, width):
        """Unit-group rows back in entity order (count, width); padding rows are zero."""
        present = [(value, rows) for value, rows in zip(values, members) if value is not None]
        stacked = concatenate([value for value, _ in present]+[Tensor(np.zeros((1, width)))])
        position = np.full(count, len(stacked.data)-1)
        if present:
            position[np.concatenate([rows for _, rows in present])] = np.arange(len(stacked.data)-1)
        return stacked[position]

    def _encode_map(self, encoded, scattered):
        """(1x1 map (map_out, n, n), map embedding): plan Â§1d."""
        p, config = self.parameters, self.config
        side = config['map_side']*SCATTER_SCALE//MAP_POOL
        real = np.zeros(len(scattered.data), dtype=bool)
        real[:len(encoded['entities'])] = True
        pixels = coarse_pixels(encoded['cells'], side*MAP_POOL)
        image = scatter_sum(scattered*Tensor(scatter_weights(pixels, real, side)[:, None]), pixels, side)
        x = self._conv(concatenate([Tensor(map_planes(encoded, MAP_POOL)), image]), 'map_project').relu()
        for stage in range(MAP_STAGES):
            x = self._conv(x, f'map_down{stage}', 2, 1).relu()
        for block in range(MAP_RESIDUAL_BLOCKS):
            residual = x
            x = self._conv(x.relu(), f'map_res{block}_a', padding=1)
            x = self._conv(x.relu(), f'map_res{block}_b', padding=1)+residual
        out = self._conv(x.relu(), 'map_out').relu()
        return out, (out.reshape((-1,)) @ p['map_fc_w']+p['map_fc_b']).relu()

    def _scalars(self, encoded):
        """Every scalar input, concatenated in scalar_input_size's order, through FC-ReLU, FC-ReLU."""
        p = self.parameters
        faction, general = encoded['player_identity']
        identity = p['base_faction_embedding'][faction]+p['general_embedding'][general]
        # Known from the lobby in fair play. (0, 0) is the unknown row.
        faction, general = encoded.get('enemy_identity', (0, 0))
        enemy = p['base_faction_embedding'][faction]+p['general_embedding'][general]
        counts = np.bincount(np.asarray(encoded['types'])[:len(encoded['entities'])], minlength=len(self.config['types']))
        opener = encoded['opener'] if 'opener' in encoded else np.zeros(len(self.config['templates'])+1)
        parts = [Tensor(visible_scalars(encoded['scalars'])), identity, enemy,
                 # Set per game by the dataset or the live actor; absent means both unrated.
                 Tensor(np.asarray(encoded.get('ratings', UNRATED), dtype=float)), Tensor(np.asarray(opener, dtype=float)),
                 Tensor(encoded['player_flags']), Tensor(np.sqrt(counts)),
                 Tensor(np.asarray(self.verb_mask(encoded), dtype=float))]
        commands = visible_task_commands(encoded['task_commands'])
        for i in range(2):  # The last command receipt, then the last validation receipt.
            template = self.type_embedding(encoded['task_command_templates'][i:i+1])[0]
            parts += [Tensor(commands[i]), template*float(encoded['task_command_template_present'][i])]
        x = (concatenate(parts) @ p['scalar_in_w']+p['scalar_in_b']).relu()
        return (x @ p['scalar_hidden_w']+p['scalar_hidden_b']).relu()

    def initial_state(self):
        """The core LSTM's (h, c) per layer at a game's start."""
        size = widths(self.config)['core']
        return [(np.zeros(size), np.zeros(size)) for _ in range(LSTM_LAYERS)]

    def _core(self, core_input, state):
        """(core output, next state): FC-ReLU u, then u + a residual LSTM step (plan Â§1e)."""
        p, size = self.parameters, widths(self.config)['core']
        u = (core_input @ p['core_in_w']+p['core_in_b']).relu()
        x, following = u, []
        for layer, (h, c) in enumerate(self.initial_state() if state is None else state):
            h, c = tensor(h), tensor(c)
            gates = ((x @ p[f'core_lstm{layer}_x_w']+h @ p[f'core_lstm{layer}_h_w']).reshape((4, size)).layer_norm()
                     * p[f'core_lstm{layer}_gate_g']+p[f'core_lstm{layer}_gate_b'])
            c = gates[1].sigmoid()*c+gates[0].sigmoid()*gates[2].tanh()
            h = gates[3].sigmoid()*self._layer_norm(c, f'core_lstm{layer}_cell').tanh()
            following.append((h, c))
            x = h
        return u+x, following

    def encode_observation(self, encoded):
        """Encoder outputs for one observation: unit features, keys, the 1x1 map and the core input."""
        p, W = self.parameters, widths(self.config)
        units, effects = self._entity_inputs(encoded)
        count, real = len(units.data), len(encoded['entities'])
        owners = [e['owner'] for e in encoded['entities']]+[-1]*(count-real)
        groups = unit_groups(encoded['own'], owners, np.arange(count) < real)
        members = [np.flatnonzero(groups == g) for g in range(UNIT_GROUPS)]
        final = self._group_transformer([units[rows] if len(rows) else None for rows in members]+[effects])
        units = final[:UNIT_GROUPS]
        def project(name, width, activation=False):
            values = [None if value is None else value @ p[f'{name}_{GROUPS[g]}_w']+p[f'{name}_{GROUPS[g]}_b']
                      for g, value in enumerate(units)]
            if activation:
                values = [None if value is None else value.relu() for value in values]
            return self._entity_order(values, members, count, width)
        spatial, map_embedding = self._encode_map(encoded, project('scatter', W['scatter'], True))
        pools = [self._pool(value, GROUPS[g]) for g, value in enumerate(final)]
        return {'features': self._entity_order(units, members, count, W['entity']),
                'keys': project('key', W['key']), 'map': spatial,
                'core_input': concatenate([self._scalars(encoded), *pools, map_embedding])}

    def _trunk(self, encoded, state=None):
        trunk = self.encode_observation(encoded)
        trunk['core'], trunk['state'] = self._core(trunk['core_input'], state)
        p = self.parameters
        trunk['value'] = (trunk['core'] @ p['value_w']+p['value_b'])[0]
        return trunk

    def _with_action(self, trunk, head, action, summary=None):
        """[core output â€– this head's embedding of the selected action â€– selected units' mean, max]."""
        parts = [trunk['core'], self.parameters[head+'_action_embedding'][action]]
        return concatenate(parts+([summary] if summary is not None else []))

    def _action_logits(self, trunk):
        p = self.parameters
        return trunk['core'] @ p['action_w']+p['action_b']

    def _delay_logits(self, trunk, action):
        p = self.parameters
        return self._with_action(trunk, 'delay', action) @ p['delay_w']+p['delay_b']

    def _lstm_step(self, query, state):
        """One step of torch's LSTM equations; state is (hidden, cell)."""
        p, key = self.parameters, len(query.data)
        gates = query @ p['unit_lstm_input_w']+state[0] @ p['unit_lstm_hidden_w']+p['unit_lstm_b']
        cell = gates[key:2*key].sigmoid()*state[1]+gates[:key].sigmoid()*gates[2*key:3*key].tanh()
        return gates[3*key:].sigmoid()*cell.tanh(), cell

    def _unit_begin(self, trunk, action):
        p, size = self.parameters, widths(self.config)['unit_query']
        zeros = Tensor(np.zeros(size))
        return {'context': self._with_action(trunk, 'unit', action) @ p['unit_in_w']+p['unit_in_b'],
                'state': (zeros, zeros), 'embedding': p['unit_action_embedding'][action]}

    def _unit_scores(self, trunk, unit):
        """(scores over every entity then the end token, next unit state): SCC's conditioned
        concat-attention, score_i = e_a . tanh(W [query; key_i])."""
        p = self.parameters
        state = self._lstm_step(unit['context'].relu() @ p['unit_query_w']+p['unit_query_b'], unit['state'])
        keys = concatenate([trunk['keys'], p['unit_end_key'].reshape((1, -1))])
        hidden = (keys @ p['unit_score_k_w']+(state[0] @ p['unit_score_q_w']+p['unit_score_b'])).tanh()
        return hidden @ unit['embedding'], {**unit, 'state': state}

    def _unit_add(self, trunk, unit, index):
        p = self.parameters
        return {**unit, 'context': unit['context']+(trunk['keys'][index]/SELECTED_UNIT_DIVISOR) @ p['unit_ar_w']+p['unit_ar_b']}

    def _selection_summary(self, trunk, selected):
        """[Mean, Max] of the selected units' final features; zeros without a selection."""
        if not selected:
            return Tensor(np.zeros(2*widths(self.config)['entity']))
        rows = trunk['features'][np.asarray(selected)]
        return concatenate([rows.mean(axis=0), rows.max(axis=0)])

    def _slot_logits(self, trunk, action, summary):
        p = self.parameters
        return self._with_action(trunk, 'slot', action, summary) @ p['slot_w']+p['slot_b']

    def _target_logits(self, trunk, action, summary):
        p = self.parameters
        query = self._with_action(trunk, 'target', action, summary) @ p['target_in_w']+p['target_in_b']
        return (trunk['keys'] @ p['target_score_k_w']+query).tanh() @ p['target_action_embedding'][action]

    def _angle_logits(self, trunk, action):
        p = self.parameters
        return self._with_action(trunk, 'angle', action) @ p['angle_w']+p['angle_b']

    def _location(self, trunk, action, summary, build_features=None):
        """Logits (n*n, offsets) for one position argument: one 256x256 map, capped as a whole."""
        p, config = self.parameters, self.config
        n, m = config['map_side'], config['map_hidden']
        query = self._with_action(trunk, 'position', action, summary) @ p['location_query_w']+p['location_query_b']
        x = self._conv(concatenate([trunk['map'], query.reshape((4, n, n))]).relu(), 'location_input').relu()
        film = query @ p['location_film_w']+p['location_film_b']
        for block in range(LOCATION_BLOCKS):
            residual = self._conv(x, f'location_block{block}_a').relu()
            y = self._conv(residual, f'location_block{block}_b', padding=1)
            gamma = film[2*block*m:(2*block+1)*m].reshape((m, 1, 1))+1
            beta = film[(2*block+1)*m:(2*block+2)*m].reshape((m, 1, 1))
            x = ((y*gamma+beta).relu()+residual).relu()
        if build_features is not None:
            x = x+(Tensor(build_features) @ p['build_map_w']).transpose().reshape((m, n, n))
        y = x
        stages = location_stages(config)
        for stage in range(len(stages)):
            y = conv_transpose2d(y, p[f'location_up{stage}'], 2, 1)+p[f'location_up{stage}_b']
            if stage < len(stages)-1:
                y = y.reshape((1, -1)).layer_norm().reshape(y.data.shape).relu()
        fine = config['position_bins']//n
        flat = y.reshape((-1,))
        flat = ((flat-flat[int(np.argmax(flat.data))])/LOGIT_CAP).tanh()*LOGIT_CAP
        return flat[location_order(config)].reshape((n*n, fine*fine))

    def _sampling_logits(self, logits):
        return logits

    def decide(self, encoded, memory=None, *, teacher=None, rng=None, deterministic=False):
        """One decision and the next core state: the selected action, its delay, then the action's arguments.

        memory is the core LSTM state from this player's previous decision (None at a game's start).
        Returns (actions, trace, next memory, value, log-probability, entropy)."""
        from event_schedule import delay_mask
        trunk = self._trunk(encoded, memory)
        config, trace = self.config, {}
        logp, entropy = Tensor(0), Tensor(0)
        def draw(name, logits, mask, override=None):
            nonlocal logp, entropy
            logits = self._sampling_logits(logits)
            if name not in ('cell', 'offset'):  # Position: capped in _location.
                logits = cap_logits(logits, mask)
            if teacher is None and name != 'delay' and not deterministic:
                logits = logits/SAMPLING_TEMPERATURE
            choice, lp, ent = categorical(logits, mask, choice=override, rng=rng, deterministic=deterministic)
            logp, entropy = logp+lp, entropy+ent
            trace[name] = choice
            return choice
        mask = self.action_mask(encoded)
        if teacher is None:  # Live play: only objects some unit can use now.
            exact = mask & self.exact_action_mask(encoded)
            mask = exact if exact.any() else mask
        action = draw('decision', self._action_logits(trunk), mask, self.action_choice(teacher) if teacher else None)
        k, obj = (int(value) for value in self.actions[action])
        kind = KINDS[k]
        delays = delay_mask(config, encoded.get('event_rate', 60))
        first_delay = int(np.flatnonzero(delays)[0])
        delay = draw('delay', self._delay_logits(trunk, action), delays, teacher.get('delay', first_delay) if teacher else None)
        if kind == 'wait':
            return [], {'kind': 'wait', 'delay': delay}, trunk['state'], trunk['value'], logp, entropy
        actors = self.actor_mask(encoded, kind)
        if teacher is None and obj >= 0:
            actors &= self.object_actors(encoded, kind, obj)
        selected, limit = [], selection_limit(config, kind)
        selection = teacher['selection'] if teacher else None
        unit = self._unit_begin(trunk, action)
        for ordinal in range(limit):
            scores, unit = self._unit_scores(trunk, unit)
            available = self.selection_mask(encoded, kind, selected, actors)
            actor_mask = np.concatenate([available, [ordinal > 0]])
            actor_mask[selected] = False
            forced = (selection[ordinal] if ordinal < len(selection) else len(actors)) if teacher else None
            index = draw('unit', scores, actor_mask, forced)
            if index == len(actors):
                break
            selected.append(index)
            unit = self._unit_add(trunk, unit, index)
            available = self.selection_mask(encoded, kind, selected, actors)
            if not any(available[i] for i in range(len(actors)) if i not in selected):
                break
        trace = {'kind': kind, 'selection': selected, 'delay': delay}
        ids = [encoded['entities'][i]['id'] for i in selected]
        result = {'kind': kind}
        if kind in GROUP_KINDS:
            result['units'] = ids
        else:
            result['builder' if kind in BUILDER_KINDS else 'producer'] = ids[0]
        if kind in COMMAND_KINDS:
            trace['command'] = obj
            result['command'] = config['commands'][obj]
        if kind in ('build', 'train'):
            trace['template'] = obj
            result['template'] = self.production_native(encoded['entities'][selected[0]], kind, config['templates'][obj])
        summary = self._selection_summary(trunk, selected)
        if kind == 'cancel_production':
            result['slot'] = draw('slot', self._slot_logits(trunk, action, summary), self.slot_mask(encoded, selected),
                                  teacher.get('slot') if teacher else None)
        mode = command_target(config, trace)
        if mode == 'entity':
            index = draw('target', self._target_logits(trunk, action, summary),
                         self.target_mask(encoded, kind, selected, trace.get('command')), teacher.get('target') if teacher else None)
            result['target'] = encoded['entities'][index]['id']
        build_mask, placement_unavailable = None, False
        if kind == 'build':
            angle = draw('angle', self._angle_logits(trunk, action), np.ones(config['angle_bins'], bool),
                         teacher.get('angle') if teacher else None)
            if config['placement_mode'] == 'full_mask':
                from build_placement import mask_for
                build_mask = mask_for(config, encoded, selected[0], obj, angle)
                placement_unavailable = not build_mask.any()
                result['angle'] = math.tau*((angle+8*encoded['rotated'])%16)/16
                if placement_unavailable:
                    trace.update(kind='wait', attempted_kind='build', placement_unavailable=True)
            else:
                result['angle'] = math.tau*angle/config['angle_bins']
                if encoded['rotated']:
                    result['angle'] = (result['angle']+math.pi)%math.tau
        if mode == 'position' and not placement_unavailable:
            bins, n = config['position_bins'], config['map_side']
            fine = bins//n
            cell_mask, offset_mask = np.ones(n*n, dtype=bool), np.ones(fine*fine, dtype=bool)
            build_features = None
            if build_mask is not None:
                from build_placement import categorical_masks, coarse_features
                build_features = coarse_features(build_mask, n)
                cell_mask, _ = categorical_masks(build_mask, n)
            logits = self._location(trunk, action, summary, build_features)
            # One 256x256 distribution, drawn as its map cell and then the offset within it.
            cell = draw('cell', logits.logsumexp(1), cell_mask, teacher.get('cell') if teacher else None)
            if build_mask is not None:
                _, offset_mask = categorical_masks(build_mask, n, cell)
            offset = draw('offset', logits[cell], offset_mask, teacher.get('offset') if teacher else None)
            x, y = (cell%n)*fine+offset%fine, (cell//n)*fine+offset//fine
            trace['x'], trace['y'] = x, y
            normalized = (np.asarray([x, y])+.5)/bins
            if encoded['rotated']:
                normalized = 1-normalized
            xy = encoded['low']+encoded['span']*normalized
            result['position'] = [float(xy[0]), float(xy[1]), 0]
        return ([] if placement_unavailable else [result]), trace, trunk['state'], trunk['value'], logp, entropy

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
