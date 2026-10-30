"""Causal owned-task inputs and observation-local entity relations."""
from __future__ import annotations

import math
import numpy as np

TASKS = ('unknown', 'idle', 'move', 'attack_move', 'attack', 'construct',
         'collect', 'return_supplies', 'dock', 'produce', 'guard', 'repair', 'other')
COMMANDS = ('unknown', 'other', 'move', 'attack_move', 'attack', 'stop', 'sell',
            'build', 'train', 'rally', 'dock', 'resume_construction',
            'upgrade', 'science', 'ability', 'enter', 'exit', 'evacuate',
            'guard', 'guard_air', 'cancel_production', 'cancel_construction', 'scatter', 'add_waypoint',
            'force_attack', 'force_attack_ground', 'repair', 'get_repaired')
RESULTS = ('unknown', 'submitted', 'accepted', 'rejected')
REASONS = ('unknown', 'none', 'unavailable_target', 'placement', 'resources', 'capability', 'invalid')
RELATIONS = ('target', 'combat_target', 'builder', 'producer', 'preferred_dock',
             'submission_target', 'validation_target')
STATUS = ('none', 'current', 'unavailable')
NUMERIC = len(TASKS) + 7 + 5 + 3 + 2 + 6
RELATION_FEATURES = len(STATUS) + 3
COMMAND_FEATURES = 1 + len(COMMANDS) + 3 + len(RESULTS) + len(REASONS) + 1
# A live-only validation receipt starts neutral until genuine examples train it.
VALIDATION_COLUMNS = (len(TASKS)+7+5+3+1, NUMERIC-3, NUMERIC-2, NUMERIC-1)
RELATION_INDEX = {name: i for i, name in enumerate(RELATIONS)}
STATUS_INDEX = {name: i for i, name in enumerate(STATUS)}


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid unit-task '+name)
    return value


def encode(config, observation, encoded, *, rotated):
    entities = encoded['entities']
    count = len(encoded['numeric'])
    numeric = np.zeros((count, NUMERIC))
    relations = np.zeros((count, len(RELATIONS)), dtype=np.int64)
    valid = np.zeros_like(relations, dtype=bool)
    relation_features = np.zeros((count, len(RELATIONS), RELATION_FEATURES))
    vocabulary = COMMANDS
    commands = np.zeros((2, COMMAND_FEATURES))
    templates = np.zeros(2, dtype=np.int64)
    template_present = np.zeros(2, dtype=bool)
    low, span = encoded['low'], encoded['span']
    types = {name: i for i, name in enumerate(config['types'])}

    def position(value):
        raw = np.asarray(value, dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError('Invalid unit-task position')
        point = (raw[:2]-low)/span
        return 1-point if rotated else point

    ids, positions = encoded['ids'], encoded['numeric']
    flagged = ([], [], [])  # (row, relation, status) one-hot cells, set together below

    def reference(row, role, ref):
        status = ref.get('status', 'unavailable')
        if status not in STATUS_INDEX:
            raise ValueError('Unknown unit-task reference status')
        target = ids.get(ref.get('id')) if status == 'current' else None
        if target is not None:
            other = entities[target]
            if not (other['own'] or other['visible']):
                target = None
            if role in ('builder', 'producer') and not other['own']:
                target = None
        if status == 'current' and target is None:
            status = 'unavailable'
        index = RELATION_INDEX[role]
        flagged[0].append(row); flagged[1].append(index); flagged[2].append(STATUS_INDEX[status])
        if target is not None:
            relations[row, index], valid[row, index] = target, True
            delta = positions[target, :2]-positions[row, :2]
            relation_features[row, index, 3:] = [*delta, float(delta @ delta)]

    contexts = [observation.get(name, {'available': False})
                for name in ('last_submission', 'last_validation')]
    for context, output, index in zip(contexts, commands, range(2)):
        if not context.get('available', False):
            continue
        kind = context.get('kind', 'unknown')
        kind = kind if kind in vocabulary else 'other'
        result = context.get('result', 'unknown')
        reason = context.get('reason', 'none' if result == 'submitted' else 'unknown')
        if result not in RESULTS or reason not in REASONS:
            raise ValueError('Unknown unit-task receipt value')
        age = _finite(context.get('age_seconds', 0), 'command age')
        template = context.get('template')
        template_present[index] = template in types
        templates[index] = types.get(template, 0)
        values = [1, *[float(kind == name) for name in vocabulary],
                  math.log1p(age)/5, math.log1p(len(context.get('actors', [])))/5,
                  float('position' in context), *[float(result == name) for name in RESULTS],
                  *[float(reason == name) for name in REASONS], float(template_present[index])]
        output[:] = values

    for i, entity in enumerate(entities):
        if not entity['own']:
            continue
        task = entity.get('task')
        if not isinstance(task, dict) or not task.get('available', False):
            raise ValueError('Missing owned-unit task snapshot')
        kind = task.get('kind', 'unknown')
        if kind not in TASKS:
            raise ValueError('Unknown unit-task category')
        destination = task.get('destination')
        point = position(destination) if destination is not None else np.zeros(2)
        delta = point-encoded['numeric'][i, :2] if destination is not None else np.zeros(2)
        cargo = task.get('cargo', {'available': False})
        supply_interface = bool(cargo.get('available', False))
        boxes = _finite(cargo.get('boxes', 0), 'cargo') if supply_interface else 0
        capacity = _finite(cargo.get('capacity', 0), 'cargo capacity') if supply_interface else 0
        if boxes < 0 or capacity < 0 or boxes > capacity:
            raise ValueError('Invalid collector cargo capacity')
        # The engine exposes SupplyTruckAIInterface on units such as Helixes
        # that cannot carry supplies. A zero-capacity interface is not cargo.
        cargo_available = supply_interface and capacity > 0
        membership = [bool(c.get('available', False) and entity['id'] in c.get('actors', [])) for c in contexts]
        requested = []
        for c, member in zip(contexts, membership):
            present = member and 'position' in c
            d = position(c['position'])-encoded['numeric'][i, :2] if present else np.zeros(2)
            requested += [float(present), *d]
        numeric[i] = [*[float(kind == name) for name in TASKS], 1,
            *[float(bool(task.get(name, False))) for name in ('moving', 'attacking', 'waiting_for_path', 'busy', 'disabled')],
            float(destination is not None), *point, *delta, float(delta @ delta),
            float(cargo_available), boxes/max(capacity, 1), float(bool(cargo.get('ferrying', False))),
            *map(float, membership), *requested]
        for role in ('target', 'combat_target', 'preferred_dock'):
            reference(i, role, task.get(role, {'status': 'none'}))
        for role in ('builder', 'producer'):
            key = role+'_id'
            reference(i, role, {'status': 'current', 'id': entity[key]} if key in entity else {'status': 'none'})
        for context, member, role in zip(contexts, membership, RELATIONS[-2:]):
            if role == 'validation_target' and not member:
                continue
            reference(i, role, context.get('target', {'status': 'none'}) if member else {'status': 'none'})
    if flagged[0]:
        relation_features[flagged] = 1
    encoded.update(task_numeric=numeric, task_relations=relations, task_relation_valid=valid,
        task_relation_features=relation_features, task_commands=commands,
        task_command_templates=templates, task_command_template_present=template_present)
    return encoded
