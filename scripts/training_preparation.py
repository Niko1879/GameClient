"""CPU-side learner preparation: microbatches, padded inputs and teacher plans.

NumPy only, so a reader process can prepare upcoming updates while the learner
process uses the GPU. Arrays carry dtype tags ('float', 'int64', 'bool'); the
learner's stager maps 'float' to its own precision.
"""
from __future__ import annotations

import numpy as np

from event_schedule import delay_mask
from generals_policy import (COMMAND_KINDS, DERIVED_ENTITY_FEATURES, KINDS, UNRATED, NeuralPolicy, command_target,
                             count_channels, derived_inputs, visible_scalars, visible_task_commands)
from training_legality import cached_legality

# Fitted Radeon encoder cost: per microbatch, per padded token, per padded pair.
MICROBATCH_COST = (12.8e-3, 2.6e-6, 1.4e-9)


class PreparationSpec:
    """What preparation needs from a learner, without its weights or device."""
    def __init__(self, config, legality_key, *, float_dtype=np.float32, device_type='cuda', reference=None):
        self.config, self.legality_key = config, legality_key
        self.float_dtype, self.device_type = np.dtype(float_dtype), device_type
        self.reference = reference or NeuralPolicy(config, parameters=False)


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
    numeric = np.zeros((batch, count, numeric_size+DERIVED_ENTITY_FEATURES), dtype=spec.float_dtype)
    types = np.zeros((batch, count), dtype=np.int64)
    queued, pixels, owners = types.copy(), types.copy(), np.full_like(types, -1)
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
        numeric[index, :length, numeric_size:] = derived_inputs(row['numeric'])
        types[index, :length], queued[index, :length] = row['types'], row['queued_types']
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
        'queued': (queued, 'int64'), 'scatter_pixels': (pixels, 'int64'),
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
        'ratings': (np.stack([np.asarray(row.get('ratings', UNRATED), dtype=spec.float_dtype) for row in encoded]), 'float')}
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
    add('entity_counts', [len(row['numeric']) for row in rows], 'float')
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
    # Player-level commands (science) select no units: a microbatch of only
    # those has no unit tokens, and the unit head is skipped.
    if unit_rows:
        categorical('unit', unit_masks, unit_choices, unit_rows)
    plan['unit_tokens'] = len(unit_rows)

    def branch(name, predicate):
        indices = np.asarray([i for i, target in enumerate(targets) if predicate(target)], dtype=np.int64)
        add(name+'_rows', indices)
        plan[name+'_rows'] = indices
        return indices

    indices = branch('science', lambda t: t['kind'] == 'science')
    if len(indices):
        categorical('science', [legal[i]['science'] for i in indices], [targets[i]['science'] for i in indices], indices)
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
        categorical('angle', np.ones((len(indices), config['angle_bins']), dtype=bool),
                    [targets[i]['angle'] for i in indices], indices)
    indices = branch('spatial', lambda t: command_target(config, t) == 'position')
    if len(indices):
        add('spatial_range', np.arange(len(indices)))
        categorical('cell', [legal[i]['cell'] for i in indices], [targets[i]['cell'] for i in indices], indices)
        categorical('offset', [legal[i]['offset'] for i in indices], [targets[i]['offset'] for i in indices], indices)
        if config['placement_mode'] == 'full_mask':
            build_spatial = [j for j, i in enumerate(indices) if targets[i]['kind'] == 'build']
            add('build_spatial', build_spatial)
            if build_spatial:
                add('build_features', np.stack([rows[indices[j]]['build_features'] for j in build_spatial]), 'float')
    return plan


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
                                'source_ticks_per_second': row.get('source_ticks_per_second', 60)} for row in micro],
                      'entity_count': numeric.shape[1]})
    return {'rows': len(rows), 'microbatches': parts}
