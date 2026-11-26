"""CPU-side learner preparation: microbatches, padded inputs and teacher plans.

NumPy only, so a reader process can prepare upcoming updates while the learner
process uses the GPU. Arrays carry dtype tags ('float', 'int64', 'bool'); the
learner's stager maps 'float' to its own precision.
"""
from __future__ import annotations

import os

import numpy as np

from generals_policy import (ARMS_FEATURES, DERIVED_ENTITY_FEATURES, EFFECT_FEATURES, GROUPS, MAP_POOL, SCATTER_SCALE, UNIT_GROUPS,
                             UNKNOWN_APM, UNRATED, NeuralPolicy, arms_inputs, coarse_pixels, command_target, count_channels, derived_inputs,
                             pool_grid, unit_groups, visible_scalars,
                             visible_task_commands)
from training_legality import cached_legality

# Modelled encoder cost: per microbatch, per padded token, per padded pair. The fixed cost is the CPU launch
# time (5090, 22 Nov 2026: 50 ms beat the Radeon fit's 12.8 ms, 24 real updates in 151 s against 163 s).
MICROBATCH_COST = (50e-3, 2.6e-6, 1.4e-9)
# Rows per microbatch, whatever the pair budget allows. The budget bounds attention (entities
# squared), but the map encoder and location upsampler grow with rows. Measured on a 5090 (22 Nov 2026,
# AlphaGenerals, 240 lanes): 512 peaks at 9.5 GB; 1,024 is no faster (the GPU is the limit by then).
# A split changes only speed: an update's loss is a sum.
MICROBATCH_ROWS = 512


class PreparationSpec:
    """What preparation needs from a learner, without its weights or device."""
    def __init__(self, config, legality_key, *, float_dtype=np.float32, device_type='cuda', reference=None):
        self.config, self.legality_key = config, legality_key
        self.float_dtype, self.device_type = np.dtype(float_dtype), device_type
        self.reference = reference or NeuralPolicy(config, parameters=False)


def order_distance(row, selected, target, bins):
    """Pixel distance from the selected units' centre to a position target; None without units.

    Entity columns 0-1 are positions normalised to the (possibly mirrored) map, the same
    frame the target pixel was quantised in.
    """
    if not len(selected):
        return None
    centre = np.asarray(row['numeric'])[np.asarray(selected), :2].mean(axis=0)*bins
    return float(np.hypot(target['x']+.5-centre[0], target['y']+.5-centre[1]))


def microbatches(rows, pair_budget, cost=MICROBATCH_COST):
    """Split one update by entity count, minimising modelled padded work.

    Each microbatch respects the padded pair budget; larger observations run
    individually. An update's loss is a sum, so grouping changes only speed.
    """
    fixed, token, pair = cost
    # A row's size bounds every group's width: its entities, or its effect tokens when there are more
    # (rare, up to 3x its entities; they would set the padding of every row they share a chunk with).
    def size(row):
        return max(len(row['encoded']['numeric']), len(row['encoded'].get('effect_numeric', ())))
    ordered = sorted(rows, key=size)
    sizes = [size(row) for row in ordered]
    best, cut = np.zeros(len(sizes)+1), np.zeros(len(sizes)+1, dtype=np.int64)
    for end in range(1, len(sizes)+1):
        largest = sizes[end-1]
        start = max(0, end-min(MICROBATCH_ROWS, max(1, pair_budget//max(largest*largest, 1))))
        costs = best[start:end]+fixed+(end-np.arange(start, end))*largest*(token+pair*largest)
        choice = int(np.argmin(costs))
        best[end], cut[end] = costs[choice], start+choice
    bounds, end = [], len(sizes)
    while end:
        bounds.append((int(cut[end]), end)); end = int(cut[end])
    for start, end in reversed(bounds):
        yield ordered[start:end]


def available_actions(spec, row):
    cached = row.get('_training_legality')
    if cached is not None and cached['key'] == spec.legality_key:
        return cached['verb']
    return spec.reference.verb_mask(row)


def visibility_shares(visibility, side, dtype):
    """(batch, 3, pooled area): each visibility category's share of the pooled pixel (map_planes)."""
    coarse = side//MAP_POOL
    grid = visibility.reshape(len(visibility), side, side)
    shares = np.zeros((len(visibility), 3, coarse, coarse), dtype=np.uint8)
    for category in range(3):
        present = (grid == category).astype(np.uint8)
        for y in range(MAP_POOL):
            for x in range(MAP_POOL):  # Strided sums: far cheaper than a sum over two block axes.
                shares[:, category] += present[:, y::MAP_POOL, x::MAP_POOL]
    return (shares.astype(dtype)/MAP_POOL**2).reshape(len(visibility), 3, coarse*coarse)


def encode_values(spec, encoded):
    """Pad only this chunk's entity dimension; padding never enters attention/pooling."""
    batch, count = len(encoded), max(len(row['numeric']) for row in encoded)
    side = spec.config['map_side']*SCATTER_SCALE  # Stored grids; the network reads them pooled (MAP_POOL).
    numeric_size = encoded[0]['numeric'].shape[1] + encoded[0]['extra_numeric'].shape[1]
    derived_end = numeric_size+DERIVED_ENTITY_FEATURES
    # Model-only switch, default off (user decision, 12 Nov 2026): without it every row is
    # unrated, so ratings_w adds nothing. Reports still band rows by row['rating'].
    # GENERALS_RATING_INPUT=1 turns it on without a config key: rating_input is not in event_store.MODEL_ONLY,
    # so the key would change the feature identity of every shard (12 Nov: with-rating arm on existing shards).
    # A checkpoint trained this way does not record it; its run directory holds a RATING_INPUT marker instead.
    rated = bool(spec.config.get('rating_input', False)) or os.environ.get('GENERALS_RATING_INPUT') == '1'
    no_opener = np.zeros(len(spec.config['templates'])+1, dtype=spec.float_dtype)
    # Columns as generals_policy._encode_modalities: stored numeric, extra, derived, then arms_inputs.
    numeric = np.zeros((batch, count, derived_end+ARMS_FEATURES), dtype=spec.float_dtype)
    types = np.zeros((batch, count), dtype=np.int64)
    pixels, owners = types.copy(), np.full_like(types, -1)
    queue_slots = encoded[0]['queue_units'].shape[1]
    # Padding entities hold empty queues: no unit (-1), no upgrade (0).
    queue_units = np.full((batch, count, queue_slots), -1, dtype=np.int64)
    queue_upgrades = np.zeros((batch, count, queue_slots), dtype=np.int64)
    # Entities outside the four count planes, including padding, land in a fifth that is discarded.
    channels = np.full_like(types, 4)
    # Rows of one game share their terrain; stage it once per game in this microbatch.
    statics, static_index, static_ids = [], np.zeros(batch, dtype=np.int64), {}
    slots = encoded[0]['entity_upgrades'].shape[1]
    upgrades = np.zeros((batch, count, slots), dtype=np.int64)
    command_slots = encoded[0]['entity_commands'].shape[1]
    commands = np.zeros((batch, count, command_slots), dtype=np.int64)
    command_states = np.zeros((batch, count, command_slots, 2), dtype=spec.float_dtype)
    valid, real, own, visible = [np.zeros((batch, count), dtype=bool) for _ in range(4)]
    for index, row in enumerate(encoded):
        length = len(row['numeric'])
        width = row['numeric'].shape[1]
        numeric[index, :length, :width] = row['numeric']
        numeric[index, :length, width:numeric_size] = row['extra_numeric']
        numeric[index, :length, numeric_size:derived_end] = derived_inputs(row['numeric'])
        numeric[index, :length, derived_end:] = arms_inputs(row)
        types[index, :length] = row['types']
        queue_units[index, :length], queue_upgrades[index, :length] = row['queue_units'], row['queue_upgrades']
        upgrades[index, :length] = row['entity_upgrades']
        commands[index, :length], command_states[index, :length] = row['entity_commands'], row['entity_command_states']
        pixels[index, :length] = coarse_pixels(row['cells'], side)
        valid[index, :length] = True  # Includes the reference's single empty-entity dummy.
        entities = row['entity_count'] if 'entity_count' in row else len(row['entities'])
        real[index, :entities] = True
        channels[index, :entities] = count_channels(row['numeric'], entities)
        terrain = row['map_static']
        if id(terrain) not in static_ids:
            static_ids[id(terrain)] = len(statics)
            statics.append(pool_grid(np.asarray(terrain).T, side).T)
        static_index[index] = static_ids[id(terrain)]
        own[index, :length] = row['own']
        cached = row.get('_training_legality')
        if cached is not None and cached['key'] == spec.legality_key:
            owners[index, :length], visible[index, :length] = cached['owners'], cached['visible']
        else:
            for entity_index, entity in enumerate(row['entities']):
                owners[index, entity_index], visible[index, entity_index] = entity['owner'], entity['visible']
    enemy = real & ~own & (owners >= 2)
    neutral = real & ~own & (owners < 2)
    values = {'numeric': (numeric, 'float'), 'types': (types, 'int64'),
        'queue_units': (queue_units, 'int64'), 'queue_upgrades': (queue_upgrades, 'int64'),
        'scatter_pixels': (pixels, 'int64'),
        'valid': (valid, 'bool'), 'real': (real, 'float'),
        'pool_own': (own, 'bool'), 'pool_visible': (enemy & visible, 'bool'),
        'pool_hidden': (enemy & ~visible, 'bool'), 'pool_neutral': (neutral, 'bool'),
        'map_channels': (channels, 'int64'),
        'map_statics': (np.stack(statics), 'float'), 'map_static_index': (static_index, 'int64'),
        'map_visibility': (visibility_shares(np.stack([row['map_visibility'] for row in encoded]), side, spec.float_dtype), 'float'),
        'scalars': (visible_scalars(np.stack([row['scalars'] for row in encoded])), 'float'),
        'entity_upgrades': (upgrades, 'int64'),
        'entity_commands': (commands, 'int64'), 'entity_command_states': (command_states, 'float'),
        'player_flags': (np.stack([row['player_flags'] for row in encoded]), 'float'),
        'available_actions': (np.stack([available_actions(spec, row) for row in encoded]), 'float'),
        'identities': ([row['player_identity'] for row in encoded], 'int64'),
        'enemy_identities': ([row.get('enemy_identity', (0, 0)) for row in encoded], 'int64'),
        'ratings': (np.stack([np.asarray(row.get('ratings', UNRATED) if rated else UNRATED, dtype=spec.float_dtype) for row in encoded]), 'float'),
        # The player's own pace (generals_policy.apm_inputs), set per row by event_dataset; absent means unknown.
        'apm': (np.stack([np.asarray(row.get('apm', UNKNOWN_APM), dtype=spec.float_dtype) for row in encoded]), 'float'),
        # Strategy input z (generals_policy.opener_inputs), set per row by event_dataset; absent means none.
        'opener': (np.stack([np.asarray(row['opener'], dtype=spec.float_dtype) if 'opener' in row else no_opener
                             for row in encoded]), 'float')}
    # v11 effect and sound-cue tokens (generals_policy.Encoder._effects), padded per chunk; padding is never a key.
    effects = max(len(row['effect_numeric']) for row in encoded)
    effect_numeric = np.zeros((batch, effects, EFFECT_FEATURES), dtype=spec.float_dtype)
    effect_types, effect_powers = np.zeros((batch, effects), dtype=np.int64), np.zeros((batch, effects), dtype=np.int64)
    effect_valid = np.zeros((batch, effects), dtype=bool)
    # Where each effect is painted when effects go on the map (network_alphastar effect_image): its pixel on
    # the entities' scatter grid; effects without a position (column 3 clear) are not painted.
    effect_pixels, effect_placed = np.zeros((batch, effects), dtype=np.int64), np.zeros((batch, effects), dtype=spec.float_dtype)
    for index, row in enumerate(encoded):
        length = len(row['effect_numeric'])
        effect_numeric[index, :length] = row['effect_numeric']
        effect_types[index, :length], effect_powers[index, :length] = row['effect_types'], row['effect_powers']
        effect_valid[index, :length] = True
        if length:
            grid = np.clip((np.asarray(row['effect_numeric'])[:, :2]*side).astype(np.int64), 0, side-1)
            effect_pixels[index, :length] = coarse_pixels(grid[:, 1]*side+grid[:, 0], side)
            effect_placed[index, :length] = np.asarray(row['effect_numeric'])[:, 3] > 0
    values.update(effect_numeric=(effect_numeric, 'float'), effect_types=(effect_types, 'int64'),
                  effect_powers=(effect_powers, 'int64'), effect_valid=(effect_valid, 'bool'),
                  effect_pixels=(effect_pixels, 'int64'), effect_placed=(effect_placed, 'float'))
    # unit_tasks fills task inputs only for owned entities; every other
    # row is zero and adds exactly nothing. Stage owned rows compactly,
    # with relation targets as flat (row*count+entity) indices.
    owned = [np.flatnonzero(row['own']) for row in encoded]
    values['task_rows'] = np.concatenate([index*count+rows for index, rows in enumerate(owned)]), 'int64'
    values['task_relations'] = np.concatenate([index*count+np.asarray(row['task_relations'])[rows]
        for index, (row, rows) in enumerate(zip(encoded, owned))]), 'int64'
    for name, dtype in (('task_numeric', 'float'), ('task_relation_valid', 'bool'),
                        ('task_relation_features', 'float')):
        values[name] = np.concatenate([np.asarray(row[name])[rows] for row, rows in zip(encoded, owned)]), dtype
    values['task_commands'] = visible_task_commands(np.stack([row['task_commands'] for row in encoded])), 'float'
    for name, dtype in (('task_command_templates', 'int64'), ('task_command_template_present', 'bool')):
        values[name] = np.stack([row[name] for row in encoded]), dtype
    # Unit groups (generals_policy.unit_groups) as gather indices into each row's tokens: its
    # entities (padded to `count`), then its effect tokens. Padding is never a group member.
    groups = unit_groups(own, owners, real)
    members = [[np.flatnonzero(groups[index] == group) for index in range(batch)] for group in range(UNIT_GROUPS)]
    members.append([count+np.arange(len(row['effect_numeric'])) for row in encoded])
    for group, rows in zip(range(len(GROUPS)), members):
        width = max(1, max(len(item) for item in rows))
        index, present = np.zeros((batch, width), dtype=np.int64), np.zeros((batch, width), dtype=bool)
        for position, item in enumerate(rows):
            index[position, :len(item)], present[position, :len(item)] = item, True
        values[f'group{group}_index'], values[f'group{group}_valid'] = (index, 'int64'), (present, 'bool')
    return values, numeric


def teacher_plan(spec, encoded, teachers, entity_count):
    """Teacher-forced head inputs: masks, choices and the rows each argument head scores.

    The selected action is one flat categorical (generals_policy.action_vocabulary) masked by
    NeuralPolicy.action_mask over the stored kind mask; every argument head reads it."""
    config, reference = spec.config, spec.reference
    masks = [cached_legality(spec, row, teacher) for row, teacher in zip(encoded, teachers)]
    values, categories = {}, {}

    def add(name, value, dtype='int64'):
        values[name] = value, dtype

    def categorical(name, mask, choices, rows=None):
        mask, choices = np.asarray(mask, dtype=bool), np.asarray(choices, dtype=np.int64)
        if (mask.ndim != 2 or choices.shape != (len(mask),) or not mask.any(axis=1).all()
                or (choices < 0).any() or (choices >= mask.shape[1]).any()
                or not mask[np.arange(len(mask)), choices].all()):
            raise ValueError('Target outside categorical mask')
        add(name+'_mask', mask, 'bool'); add(name+'_choices', choices)
        if rows is not None:
            global_rows = command_rows[np.asarray(rows, dtype=int)]
            add(name+'_global', global_rows)
            categories[name] = global_rows

    actions = [reference.action_choice(t) for t in teachers]
    categorical('decision', [reference.action_mask(row, mask['verb']) for row, mask in zip(encoded, masks)], actions)
    add('actions', actions)
    command_rows = np.asarray([i for i, target in enumerate(teachers) if target['kind'] != 'wait'], dtype=np.int64)
    add('command_rows', command_rows)
    plan = {'values': values, 'categories': categories, 'command_rows': command_rows,
            'command_count': len(command_rows), 'batch': len(teachers), 'unit_tokens': 0}
    for name in ('slot', 'target', 'angle', 'spatial'):
        plan[name+'_rows'] = np.zeros(0, dtype=np.int64)
    # The encoder's [Mean, Max] summary reads every command row's selection.
    selected = [masks[i]['selected'] for i in command_rows]
    plan['selected'] = selected
    longest = max([1, *map(len, selected)])
    selected_indices = np.zeros((len(command_rows), longest), dtype=np.int64)
    for i, selection in enumerate(selected):
        selected_indices[i, :len(selection)] = selection
    add('selected_indices', selected_indices)
    add('selected_lengths', list(map(len, selected)))
    if not len(command_rows):
        return plan
    rows = [encoded[i] for i in command_rows]
    targets = [teachers[i] for i in command_rows]
    legal = [masks[i] for i in command_rows]
    unit_rows, unit_prefixes, unit_masks, unit_choices = [], [], [], []
    for i, (row, entry) in enumerate(zip(rows, legal)):
        real_count = len(row['numeric'])
        for ordinal, (mask, choice) in enumerate(zip(entry['unit_masks'], entry['unit_choices'])):
            padded = np.zeros(entity_count+1, dtype=bool)
            padded[:real_count], padded[-1] = mask[:-1], mask[-1]
            unit_rows.append(i); unit_prefixes.append(ordinal); unit_masks.append(padded)
            unit_choices.append(entity_count if choice == real_count else choice)
    add('command_range', np.arange(len(rows)))
    add('unit_rows', unit_rows); add('unit_prefixes', unit_prefixes)
    # A microbatch without unit tokens skips the unit head.
    if unit_rows:
        categorical('unit', unit_masks, unit_choices, unit_rows)
    plan['unit_tokens'] = len(unit_rows)

    def branch(name, predicate):
        indices = np.asarray([i for i, target in enumerate(targets) if predicate(target)], dtype=np.int64)
        add(name+'_rows', indices)
        plan[name+'_rows'] = indices
        return indices

    indices = branch('slot', lambda t: t['kind'] == 'cancel_production')
    if len(indices):
        categorical('slot', [legal[i]['slot'] for i in indices], [targets[i]['slot'] for i in indices], indices)
    indices = branch('target', lambda t: command_target(config, t) == 'entity')
    if len(indices):
        padded = np.zeros((len(indices), entity_count), dtype=bool)
        for j, i in enumerate(indices):
            padded[j, :len(legal[i]['target'])] = legal[i]['target']
        categorical('target', padded, [targets[i]['target'] for i in indices], indices)
    indices = branch('angle', lambda t: t['kind'] == 'build')
    if len(indices):
        mask, choices = np.ones((len(indices), config['angle_bins']), dtype=bool), [targets[i]['angle'] for i in indices]
        categorical('angle', mask, choices, indices)
    indices = branch('spatial', lambda t: command_target(config, t) == 'position')
    if len(indices):
        add('spatial_range', np.arange(len(indices)))
        categorical('cell', [legal[i]['cell'] for i in indices], [targets[i]['cell'] for i in indices], indices)
        categorical('offset', [legal[i]['offset'] for i in indices], [targets[i]['offset'] for i in indices], indices)
        # Order lengths (pixels, None without units): the validation report's
        # distance-to-click bands (train_events.position_report).
        distances = [order_distance(rows[i], selected[i], targets[i], config['position_bins']) for i in indices]
        plan['order_distances'] = distances
        if config['placement_mode'] == 'full_mask':
            build_spatial = [j for j, i in enumerate(indices) if targets[i]['kind'] == 'build']
            add('build_spatial', build_spatial)
            if build_spatial:
                add('build_features', np.stack([rows[indices[j]]['build_features'] for j in build_spatial]), 'float')
    return plan


def head_counts(plan):
    """Scored tokens per row and head: the loss averages each argument head over its tokens."""
    batch = plan['batch']
    counts = {'decision': np.ones(batch, dtype=int)}
    for name, rows in plan['categories'].items():
        if name != 'decision':
            np.add.at(counts.setdefault(name, np.zeros(batch, dtype=int)), rows, 1)
    return counts


def acting_general(row):
    """The acting player's general ('vanilla', 'tank', 'nuke', ...), for reporting only (train_events by_general)."""
    from generals_identity import GENERALS
    identity = row.get('encoded', {}).get('player_identity')
    return GENERALS[identity[1]] if identity is not None else '<unknown>'


# A delay shorter than one canonical tick (same-tick orders) counts as one tick: the rate stays finite.
MIN_DELAY_TICKS = 1


def pace_plan(plan, rows):
    """network='paced': each row's own order rate (per second; the play default when unknown) and the pace head's
    rows (orders with a known own pace) with their delay to the next order in seconds and whether it was observed
    (False: at least that long; event_dataset.delay_targets)."""
    from generals_policy import PLAY_APM
    rate = np.asarray([(row.get('apm') or PLAY_APM)/60 for row in rows])
    scored = [i for i, row in enumerate(rows) if row.get('delay') and row.get('apm')]
    delay = np.asarray([max(rows[i]['delay'][0], MIN_DELAY_TICKS)/60 for i in scored], dtype=float)
    observed = np.asarray([float(rows[i]['delay'][1]) for i in scored], dtype=float)
    plan['values'].update(order_rate=(rate, 'float'), pace_rows=(np.asarray(scored, dtype=np.int64), 'int64'),
                          pace_delay=(delay, 'float'), pace_observed=(observed, 'float'))
    if scored:
        plan['categories']['pace'] = np.asarray(scored, dtype=np.int64)


def prepare_update(spec, rows, pair_budget):
    """Everything an update needs from the CPU, independent of the learner's device."""
    parts = []
    for micro in microbatches(rows, pair_budget):
        encoded = [row['encoded'] for row in micro]
        values, numeric = encode_values(spec, encoded)
        teachers = [row['target'] for row in micro]
        plan = teacher_plan(spec, encoded, teachers, numeric.shape[1])
        if spec.config.get('network') == 'paced':
            pace_plan(plan, micro)
        parts.append({'values': values, 'numeric': numeric,
                      'plan': plan,
                      # Sequence training: each row's lane and step in its update (plan §2).
                      'lanes': np.asarray([row.get('lane', -1) for row in micro], dtype=np.int64),
                      'steps': np.asarray([row.get('step', -1) for row in micro], dtype=np.int64),
                      'rows': [{'target': row['target'], 'frame': row.get('frame'),
                                'source_ticks_per_second': row.get('source_ticks_per_second', 60),
                                'rating': row.get('rating'), 'match_key': row.get('match_key'),
                                'player': row.get('player'), 'general': acting_general(row),
                                'opener': row.get('opener')} for row in micro],
                      'entity_count': numeric.shape[1]})
    return {'rows': len(rows), 'microbatches': parts}
