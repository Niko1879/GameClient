"""Teacher-prefix legality masks, frozen beside each prepared event."""
from __future__ import annotations

import numpy as np

from generals_policy import COMMAND_KINDS, command_target, selection_limit


def build_legality(policy, encoded, teacher, key):
    count = len(encoded['numeric'])
    config = policy.config
    result = {'key': key, 'teacher': teacher,
              'owners': np.asarray([e['owner'] for e in encoded['entities']] or [-1], dtype=np.int64),
              'visible': np.asarray([e['visible'] for e in encoded['entities']] or [False], dtype=bool),
              'verb': policy.verb_mask(encoded)}
    kind = teacher['kind']
    if kind == 'wait':
        return result
    if kind == 'science':
        # Player-level: no actor tokens, only the purchasable promotions.
        result.update(actor=np.zeros(count, dtype=bool), unit_masks=np.zeros((0, count+1), dtype=bool),
                      unit_choices=[], selected=[], science=policy.science_mask(encoded))
        return result
    actors = policy.actor_mask(encoded, kind)
    limit = selection_limit(config, kind)
    selected, unit_masks, choices = [], [], []
    for ordinal in range(limit):
        mask = policy.selection_mask(encoded, kind, selected, actors)
        mask = np.concatenate([mask, [ordinal > 0]])
        mask[selected] = False
        choice = teacher['selection'][ordinal] if ordinal < len(teacher['selection']) else count
        if choice < 0 or choice >= len(mask) or not mask[choice]:
            raise ValueError('Target outside categorical mask')
        unit_masks.append(mask); choices.append(choice)
        if choice == count:
            break
        selected.append(choice)
        available = policy.selection_mask(encoded, kind, selected, actors)
        if not any(available[i] for i in range(len(available)) if i not in selected):
            break
    result.update(actor=actors, unit_masks=np.stack(unit_masks), unit_choices=choices, selected=selected)
    if kind in COMMAND_KINDS:
        result['command'] = policy.command_mask(encoded, kind, selected)
    if kind == 'cancel_production':
        result['slot'] = policy.slot_mask(encoded, selected)
    mode = command_target(config, teacher)
    if mode == 'entity':
        result['target'] = policy.target_mask(encoded, kind, selected, teacher.get('command'))
    if kind in ('build', 'train'):
        actor = encoded['entities'][selected[0]]
        menu = config['menus'][actor['type']+'|'+kind]
        available = actor['buildable_templates' if kind == 'build' else 'trainable_templates']
        result['template'] = np.asarray([name in menu and name in available for name in config['templates']], dtype=bool)
    if mode == 'position':
        n, fine = config['map_side'], config['position_bins']//config['map_side']
        cells, offsets = np.ones(n*n, dtype=bool), np.ones(fine*fine, dtype=bool)
        if kind == 'build' and config['placement_mode'] == 'full_mask':
            from build_placement import mask_for, categorical_masks
            mask = mask_for(config, encoded, selected[0], teacher['template'], teacher['angle'])
            cells, offsets = categorical_masks(mask, n, teacher['cell'])
        result.update(cell=cells, offset=offsets)
    return result


def cached_legality(model, encoded, teacher):
    cached = encoded.get('_training_legality')
    if cached is not None and cached['key'] == model.legality_key and cached['teacher'] == teacher:
        return cached
    return build_legality(model.reference, encoded, teacher, model.legality_key)
