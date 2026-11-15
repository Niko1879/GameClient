"""CPU-side learner preparation: microbatches, padded inputs and teacher plans.

NumPy only, so a reader process can prepare upcoming updates while the learner
process uses the GPU. Arrays carry dtype tags ('float', 'int64', 'bool'); the
learner's stager maps 'float' to its own precision.
"""
from __future__ import annotations

import os

import numpy as np

from event_schedule import delay_mask
from generals_policy import (ARMS_FEATURES, COMMAND_KINDS, DERIVED_ENTITY_FEATURES, EFFECT_FEATURES, KINDS, UNRATED, NeuralPolicy, command_target,
                             arms_inputs, count_channels, derived_inputs, visible_scalars, visible_task_commands)
from training_legality import cached_legality

# Fitted Radeon encoder cost: per microbatch, per padded token, per padded pair.
MICROBATCH_COST = (12.8e-3, 2.6e-6, 1.4e-9)


class PreparationSpec:
    """What preparation needs from a learner, without its weights or device."""
    def __init__(self, config, legality_key, *, float_dtype=np.float32, device_type='cuda', reference=None):
        self.config, self.legality_key = config, legality_key
        self.float_dtype, self.device_type = np.dtype(float_dtype), device_type
        self.reference = reference or NeuralPolicy(config, parameters=False)


def validate_smoothing(smoothing):
    """config['target_smoothing'] (optional): Gaussian widths that let near misses cost little.

    {'delay': ticks, 'angle': angle bins, 'position': {kind or 'default': grid pixels}}; any
    part may be left out (exact targets there). Position widths are in position_bins pixels
    (about 10 world units on a 2,700-unit map); a kind without a width, or 0, stays exact.
    A position width may instead scale with the order's length,
    {'per_pixel': k, 'min': a, 'max': b}: clamp(k*d, a, b), where d is the pixel distance from
    the selected units' centre to the target (a when nothing is selected).
    """
    if not isinstance(smoothing, dict) or not set(smoothing) <= {'delay', 'angle', 'position'}:
        raise ValueError('target_smoothing takes delay, angle and position')
    positions = smoothing.get('position', {})
    if not isinstance(positions, dict):
        raise ValueError('target_smoothing position widths are per kind')
    widths = [smoothing.get('delay', 0), smoothing.get('angle', 0)]
    for width in positions.values():
        if isinstance(width, dict):
            if set(width) != {'per_pixel', 'min', 'max'}:
                raise ValueError('A distance-scaled width takes per_pixel, min and max')
            widths += list(width.values())
            if all(isinstance(w, (int, float)) for w in width.values()) and width['min'] > width['max']:
                raise ValueError('A distance-scaled width needs min <= max')
        else:
            widths.append(width)
    if not all(isinstance(w, (int, float)) and not isinstance(w, bool) and np.isfinite(w) and 0 <= w <= 16
               for w in widths):
        raise ValueError('target_smoothing widths must be numbers in [0, 16]')


def position_width(width, distance):
    """A kind's Gaussian width in pixels: fixed, or clamp(per_pixel*distance, min, max)."""
    if not isinstance(width, dict):
        return width
    if distance is None:
        return width['min']
    return min(max(width['per_pixel']*distance, width['min']), width['max'])


def order_distance(row, selected, target, bins):
    """Pixel distance from the selected units' centre to a position target; None without units.

    Entity columns 0-1 are positions normalised to the (possibly mirrored) map, the same
    frame the target pixel was quantised in.
    """
    if not len(selected):
        return None
    centre = np.asarray(row['numeric'])[np.asarray(selected), :2].mean(axis=0)*bins
    return float(np.hypot(target['x']+.5-centre[0], target['y']+.5-centre[1]))


def soft_targets(mask, choices, sigma, values, period=None):
    """(rows, classes) Gaussian target weights around each choice, over legal classes only."""
    mask, values = np.asarray(mask, dtype=bool), np.asarray(values, dtype=np.float64)
    distance = np.abs(values[None, :]-values[np.asarray(choices)][:, None])
    if period is not None:
        distance = np.minimum(distance, period-distance)
    weights = np.exp(-.5*(distance/sigma)**2)*mask
    weights[weights < 1e-4] = 0  # Beyond ~4.3 sigma: nothing worth a log-probability gather.
    return weights/weights.sum(axis=1, keepdims=True)


def soft_positions(targets, legal_cells, widths, config, exact, distances=None):
    """Sparse Gaussian targets on the position grid: (row, cell, offset, weight) arrays.

    Rows in `exact` (or whose kind has no width) keep their single teacher pixel. The cell
    and offset terms of the joint 256x256 distribution are scored from these entries.
    `distances` (per row, pixels or None) sizes distance-scaled widths.
    """
    n, bins = config['map_side'], config['position_bins']
    fine = bins//n
    rows, cells, offsets, weights = [], [], [], []
    for row, (target, legal) in enumerate(zip(targets, legal_cells)):
        sigma = position_width(widths.get(target['kind'], widths.get('default', 0)),
                               None if distances is None else distances[row])
        if row in exact or not sigma:
            points, w = np.array([[target['x'], target['y']]]), np.ones(1)
        else:
            radius = int(np.ceil(4.3*sigma))
            dx, dy = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
            points = np.stack([target['x']+dx.ravel(), target['y']+dy.ravel()], axis=1)
            w = np.exp(-.5*((dx.ravel()**2+dy.ravel()**2)/sigma**2))
            inside = (points >= 0).all(axis=1) & (points < bins).all(axis=1) & (w >= 1e-4)
            points, w = points[inside], w[inside]
            cell = (points[:, 1]//fine)*n+points[:, 0]//fine
            keep = np.asarray(legal, dtype=bool)[cell]
            points, w = points[keep], w[keep]
        x, y = points[:, 0], points[:, 1]
        rows.append(np.full(len(w), row)); cells.append((y//fine)*n+x//fine)
        offsets.append((y % fine)*fine+x % fine); weights.append(w/w.sum())
    return np.concatenate(rows), np.concatenate(cells), np.concatenate(offsets), np.concatenate(weights)


def microbatches(rows, pair_budget, cost=MICROBATCH_COST):
    """Split one update by entity count, minimising modelled padded work.

    Each microbatch respects the padded pair budget; larger observations run
    individually. An update's loss is a sum, so grouping changes only speed.
    """
    fixed, token, pair = cost
    ordered = sorted(rows, key=lambda row: len(row['encoded']['numeric']))
    sizes = [len(row['encoded']['numeric']) for row in ordered]
    best, cut = np.zeros(len(sizes)+1), np.zeros(len(sizes)+1, dtype=np.int64)
    for end in range(1, len(sizes)+1):
        largest = sizes[end-1]
        start = max(0, end-max(1, pair_budget//max(largest*largest, 1)))
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


def encode_values(spec, encoded):
    """Pad only this chunk's entity dimension; padding never enters attention/pooling."""
    batch, count = len(encoded), max(len(row['numeric']) for row in encoded)
    numeric_size = encoded[0]['numeric'].shape[1] + encoded[0]['extra_numeric'].shape[1]
    derived_end = numeric_size+DERIVED_ENTITY_FEATURES
    # Model-only switch, default off (user decision, 12 Nov 2026): without it every row is
    # unrated, so ratings_w adds nothing. Reports still band rows by row['rating'].
    # GENERALS_RATING_INPUT=1 turns it on without a config key: rating_input is not in event_store.MODEL_ONLY,
    # so the key would change the feature identity of every shard (12 Nov: with-rating arm on existing shards).
    # A checkpoint trained this way does not record it; its run directory holds a RATING_INPUT marker instead.
    rated = bool(spec.config.get('rating_input', False)) or os.environ.get('GENERALS_RATING_INPUT') == '1'
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
        pixels[index, :length] = row['cells']
        valid[index, :length] = True  # Includes the reference's single empty-entity dummy.
        entities = row['entity_count'] if 'entity_count' in row else len(row['entities'])
        real[index, :entities] = True
        channels[index, :entities] = count_channels(row['numeric'], entities)
        terrain = row['map_static']
        if id(terrain) not in static_ids:
            static_ids[id(terrain)] = len(statics)
            statics.append(terrain)
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
        'map_visibility': (np.stack([row['map_visibility'] for row in encoded]).astype(np.int64), 'int64'),
        'scalars': (visible_scalars(np.stack([row['scalars'] for row in encoded])), 'float'),
        'entity_upgrades': (upgrades, 'int64'),
        'entity_commands': (commands, 'int64'), 'entity_command_states': (command_states, 'float'),
        'player_flags': (np.stack([row['player_flags'] for row in encoded]), 'float'),
        'available_actions': (np.stack([available_actions(spec, row) for row in encoded]), 'float'),
        'identities': ([row['player_identity'] for row in encoded], 'int64'),
        'enemy_identities': ([row.get('enemy_identity', (0, 0)) for row in encoded], 'int64'),
        'ratings': (np.stack([np.asarray(row.get('ratings', UNRATED) if rated else UNRATED, dtype=spec.float_dtype) for row in encoded]), 'float')}
    # v11 effect and sound-cue tokens (generals_policy.Encoder._effects), padded per chunk; padding is never a key.
    effects = max(len(row['effect_numeric']) for row in encoded)
    effect_numeric = np.zeros((batch, effects, EFFECT_FEATURES), dtype=spec.float_dtype)
    effect_types, effect_powers = np.zeros((batch, effects), dtype=np.int64), np.zeros((batch, effects), dtype=np.int64)
    effect_valid = np.zeros((batch, effects), dtype=bool)
    for index, row in enumerate(encoded):
        length = len(row['effect_numeric'])
        effect_numeric[index, :length] = row['effect_numeric']
        effect_types[index, :length], effect_powers[index, :length] = row['effect_types'], row['effect_powers']
        effect_valid[index, :length] = True
    values.update(effect_numeric=(effect_numeric, 'float'), effect_types=(effect_types, 'int64'),
                  effect_powers=(effect_powers, 'int64'), effect_valid=(effect_valid, 'bool'))
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
    return values, numeric


def teacher_plan(spec, encoded, teachers, entity_count):
    config = spec.config
    masks = [cached_legality(spec, row, teacher) for row, teacher in zip(encoded, teachers)]
    verbs = np.stack([mask['verb'] for mask in masks])
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

    categorical('decision', verbs, [KINDS.index(t['kind']) for t in teachers])
    delay_masks = [delay_mask(config, row.get('event_rate', 60)) for row in encoded]
    choices = [t.get('delay', int(np.flatnonzero(mask)[0])) for t, mask in zip(teachers, delay_masks)]
    categorical('event_delay', delay_masks, choices)
    smoothing = config.get('target_smoothing') or {}
    if smoothing.get('delay'):
        add('event_delay_soft', soft_targets(delay_masks, choices, smoothing['delay'], config['delay_ticks']), 'float')
    add('delay_known', ['delay' in t for t in teachers], 'float')
    add('all_verbs', [KINDS.index(t['kind']) for t in teachers])
    add('conditional_mask', verbs[:, 1:], 'bool')
    command_rows = np.asarray([i for i, target in enumerate(teachers) if target['kind'] != 'wait'], dtype=np.int64)
    add('command_rows', command_rows)
    plan = {'values': values, 'categories': categories, 'command_rows': command_rows,
            'command_count': len(command_rows), 'batch': len(teachers)}
    if not len(command_rows):
        return plan
    rows = [encoded[i] for i in command_rows]
    targets = [teachers[i] for i in command_rows]
    legal = [masks[i] for i in command_rows]
    func = np.zeros((len(rows), len(config['types'])), dtype=np.float32)
    for i, (row, entry) in enumerate(zip(rows, legal)):
        func[i, np.asarray(row['types'])[np.asarray(entry['actor'], dtype=bool)]] = 1
    add('func_types', func, 'float')
    selected = [entry['selected'] for entry in legal]
    plan['selected'] = selected
    longest = max(1, max(map(len, selected)))
    selected_indices = np.zeros((len(rows), longest), dtype=np.int64)
    unit_rows, unit_prefixes, unit_masks, unit_choices = [], [], [], []
    for i, (row, entry, selection) in enumerate(zip(rows, legal, selected)):
        selected_indices[i, :len(selection)] = selection
        real_count = len(row['numeric'])
        for ordinal, (mask, choice) in enumerate(zip(entry['unit_masks'], entry['unit_choices'])):
            padded = np.zeros(entity_count+1, dtype=bool)
            padded[:real_count], padded[-1] = mask[:-1], mask[-1]
            unit_rows.append(i); unit_prefixes.append(ordinal); unit_masks.append(padded)
            unit_choices.append(entity_count if choice == real_count else choice)
    add('selected_indices', selected_indices)
    add('command_range', np.arange(len(rows)))
    add('selected_lengths', list(map(len, selected)))
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

    # 'command_rows' already names the non-DEFER rows; this head's rows use another key.
    indices = np.asarray([i for i, target in enumerate(targets) if target['kind'] in COMMAND_KINDS], dtype=np.int64)
    add('command_head_rows', indices)
    plan['command_head_rows'] = indices
    if len(indices):
        categorical('command', [legal[i]['command'] for i in indices], [targets[i]['command'] for i in indices], indices)
    indices = branch('slot', lambda t: t['kind'] == 'cancel_production')
    if len(indices):
        categorical('slot', [legal[i]['slot'] for i in indices], [targets[i]['slot'] for i in indices], indices)
    indices = branch('target', lambda t: command_target(config, t) == 'entity')
    if len(indices):
        padded = np.zeros((len(indices), entity_count), dtype=bool)
        for j, i in enumerate(indices):
            padded[j, :len(legal[i]['target'])] = legal[i]['target']
        categorical('target', padded, [targets[i]['target'] for i in indices], indices)
    indices = branch('template', lambda t: t['kind'] in ('build', 'train'))
    if len(indices):
        categorical('template', [legal[i]['template'] for i in indices],
                    [targets[i]['template'] for i in indices], indices)
    indices = branch('angle', lambda t: t['kind'] == 'build')
    if len(indices):
        mask, choices = np.ones((len(indices), config['angle_bins']), dtype=bool), [targets[i]['angle'] for i in indices]
        categorical('angle', mask, choices, indices)
        if smoothing.get('angle'):
            bins = config['angle_bins']
            add('angle_soft', soft_targets(mask, choices, smoothing['angle'], np.arange(bins), period=bins), 'float')
    indices = branch('spatial', lambda t: command_target(config, t) == 'position')
    if len(indices):
        add('spatial_range', np.arange(len(indices)))
        categorical('cell', [legal[i]['cell'] for i in indices], [targets[i]['cell'] for i in indices], indices)
        categorical('offset', [legal[i]['offset'] for i in indices], [targets[i]['offset'] for i in indices], indices)
        # Order lengths (pixels, None without units): distance-scaled widths and the
        # validation report's distance-to-click bands (train_events.position_report).
        distances = [order_distance(rows[i], selected[i], targets[i], config['position_bins']) for i in indices]
        plan['order_distances'] = distances
        if smoothing.get('position'):
            # Full-mask builds know legal offsets only inside the teacher's cell: kept exact.
            exact = {j for j, i in enumerate(indices)
                     if config['placement_mode'] == 'full_mask' and targets[i]['kind'] == 'build'}
            entries = soft_positions([targets[i] for i in indices], [legal[i]['cell'] for i in indices],
                                     smoothing['position'], config, exact, distances)
            for name, value, dtype in zip(('rows', 'cells', 'offsets', 'weights'), entries,
                                          ('int64', 'int64', 'int64', 'float')):
                add('position_soft_'+name, value, dtype)
            add('position_soft_exact', [j in exact for j in range(len(indices))], 'bool')
        if config['placement_mode'] == 'full_mask':
            build_spatial = [j for j, i in enumerate(indices) if targets[i]['kind'] == 'build']
            add('build_spatial', build_spatial)
            if build_spatial:
                add('build_features', np.stack([rows[indices[j]]['build_features'] for j in build_spatial]), 'float')
    return plan


def acting_general(row):
    """The acting player's general ('vanilla', 'tank', 'nuke', ...), for reporting only (train_events by_general)."""
    from generals_identity import GENERALS
    identity = row.get('encoded', {}).get('player_identity')
    return GENERALS[identity[1]] if identity is not None else '<unknown>'


def prepare_update(spec, rows, pair_budget):
    """Everything an update needs from the CPU, independent of the learner's device."""
    parts = []
    for micro in microbatches(rows, pair_budget):
        encoded = [row['encoded'] for row in micro]
        values, numeric = encode_values(spec, encoded)
        teachers = [row['target'] for row in micro]
        parts.append({'values': values, 'numeric': numeric,
                      'plan': teacher_plan(spec, encoded, teachers, numeric.shape[1]),
                      'rows': [{'target': row['target'], 'frame': row.get('frame'),
                                'source_ticks_per_second': row.get('source_ticks_per_second', 60),
                                'rating': row.get('rating'), 'match_key': row.get('match_key'),
                                'player': row.get('player'), 'general': acting_general(row)} for row in micro],
                      'entity_count': numeric.shape[1]})
    return {'rows': len(rows), 'microbatches': parts}
