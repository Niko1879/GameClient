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


def _one_hot(vocabulary):
    return {name: [float(name == other) for other in vocabulary] for name in vocabulary}


TASK_ONE_HOT, COMMAND_ONE_HOT = _one_hot(TASKS), _one_hot(COMMANDS)
RESULT_ONE_HOT, REASON_ONE_HOT = _one_hot(RESULTS), _one_hot(REASONS)
STATUS_NONE = STATUS_INDEX['none']
OWN_ROLES = tuple((RELATION_INDEX[role], role) for role in ('target', 'combat_target', 'preferred_dock'))
LINK_ROLES = tuple((RELATION_INDEX[role], role+'_id') for role in ('builder', 'producer'))
SUBMISSION, VALIDATION = RELATION_INDEX['submission_target'], RELATION_INDEX['validation_target']
NO_TARGET = {'status': 'none'}


def encode(config, observation, encoded, *, rotated, types=None):
    entities = encoded['entities']
    count = len(encoded['numeric'])
    numeric = np.zeros((count, NUMERIC))
    relations = np.zeros((count, len(RELATIONS)), dtype=np.int64)
    valid = np.zeros_like(relations, dtype=bool)
    relation_features = np.zeros((count, len(RELATIONS), RELATION_FEATURES))
    commands = np.zeros((2, COMMAND_FEATURES))
    templates = np.zeros(2, dtype=np.int64)
    template_present = np.zeros(2, dtype=bool)
    (low_x, low_y), (span_x, span_y) = encoded['low'].tolist(), encoded['span'].tolist()
    if types is None:
        types = {name: i for i, name in enumerate(config['types'])}

    def position(value):
        # The same float64 operations as (value[:2]-low)/span, mirrored when rotated.
        try:
            x, y, z = value
            finite = math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError('Invalid unit-task position')
        x, y = (x-low_x)/span_x, (y-low_y)/span_y
        return (1-x, 1-y) if rotated else (x, y)

    ids = encoded['ids']
    positions = encoded['numeric'][:, :2].tolist()
    flagged = ([], [], [])  # (row, relation, status) one-hot cells, set together below
    linked = ([], [], [], [])  # (row, relation, target, [dx, dy, squared distance])

    def reference(row, index, builder_role, status, identity):
        if status not in STATUS_INDEX:
            raise ValueError('Unknown unit-task reference status')
        target = ids.get(identity) if status == 'current' else None
        if target is not None:
            other = entities[target]
            if not (other['own'] or other['visible']):
                target = None
            elif builder_role and not other['own']:
                target = None
        if status == 'current' and target is None:
            status = 'unavailable'
        flagged[0].append(row); flagged[1].append(index); flagged[2].append(STATUS_INDEX[status])
        if target is not None:
            mine, theirs = positions[row], positions[target]
            dx, dy = theirs[0]-mine[0], theirs[1]-mine[1]
            linked[0].append(row); linked[1].append(index); linked[2].append(target)
            linked[3].append((dx, dy, dx*dx+dy*dy))

    contexts = [observation.get(name, {'available': False})
                for name in ('last_submission', 'last_validation')]
    for context, output, index in zip(contexts, commands, range(2)):
        if not context.get('available', False):
            continue
        kind = context.get('kind', 'unknown')
        result = context.get('result', 'unknown')
        reason = context.get('reason', 'none' if result == 'submitted' else 'unknown')
        if result not in RESULT_ONE_HOT or reason not in REASON_ONE_HOT:
            raise ValueError('Unknown unit-task receipt value')
        age = _finite(context.get('age_seconds', 0), 'command age')
        template = context.get('template')
        present = template in types
        template_present[index] = present
        templates[index] = types.get(template, 0)
        output[:] = [1, *COMMAND_ONE_HOT.get(kind, COMMAND_ONE_HOT['other']),
                     math.log1p(age)/5, math.log1p(len(context.get('actors', [])))/5,
                     float('position' in context), *RESULT_ONE_HOT[result],
                     *REASON_ONE_HOT[reason], float(present)]

    # Membership for each context, so a large group does not cost a scan per unit.
    actors = [set(c.get('actors', ())) if c.get('available', False) else () for c in contexts]
    requests = [position(c['position']) if actors[n] and 'position' in c else None for n, c in enumerate(contexts)]
    targets = [c.get('target', NO_TARGET) for c in contexts]
    rows, row_index = [], []
    for i, entity in enumerate(entities):
        if not entity['own']:
            continue
        task = entity.get('task')
        if not isinstance(task, dict) or not task.get('available', False):
            raise ValueError('Missing owned-unit task snapshot')
        one_hot = TASK_ONE_HOT.get(task.get('kind', 'unknown'))
        if one_hot is None:
            raise ValueError('Unknown unit-task category')
        destination = task.get('destination')
        here_x, here_y = positions[i]
        if destination is not None:
            point_x, point_y = position(destination)
            delta_x, delta_y = point_x-here_x, point_y-here_y
        else:
            point_x = point_y = delta_x = delta_y = 0.0
        cargo = task.get('cargo', NO_TARGET)
        supply_interface = bool(cargo.get('available', False))
        boxes = _finite(cargo.get('boxes', 0), 'cargo') if supply_interface else 0
        capacity = _finite(cargo.get('capacity', 0), 'cargo capacity') if supply_interface else 0
        if boxes < 0 or capacity < 0 or boxes > capacity:
            raise ValueError('Invalid collector cargo capacity')
        identity = entity['id']
        member_0, member_1 = identity in actors[0], identity in actors[1]
        # The engine exposes SupplyTruckAIInterface on units such as Helixes
        # that cannot carry supplies. A zero-capacity interface is not cargo.
        row = [*one_hot, 1,
               float(bool(task.get('moving', False))), float(bool(task.get('attacking', False))),
               float(bool(task.get('waiting_for_path', False))), float(bool(task.get('busy', False))),
               float(bool(task.get('disabled', False))),
               float(destination is not None), point_x, point_y, delta_x, delta_y, delta_x*delta_x+delta_y*delta_y,
               float(supply_interface and capacity > 0), boxes/max(capacity, 1), float(bool(cargo.get('ferrying', False))),
               float(member_0), float(member_1)]
        for member, request in ((member_0, requests[0]), (member_1, requests[1])):
            if member and request is not None:
                row += (1.0, request[0]-here_x, request[1]-here_y)
            else:
                row += (0.0, 0.0, 0.0)
        rows.append(row); row_index.append(i)
        for index, role in OWN_ROLES:
            ref = task.get(role, NO_TARGET)
            status = ref.get('status', 'unavailable')
            if status == 'none':
                flagged[0].append(i); flagged[1].append(index); flagged[2].append(STATUS_NONE)
            else:
                reference(i, index, False, status, ref.get('id'))
        for index, key in LINK_ROLES:
            if key in entity:
                reference(i, index, True, 'current', entity[key])
            else:
                flagged[0].append(i); flagged[1].append(index); flagged[2].append(STATUS_NONE)
        if member_0:
            ref = targets[0]
            reference(i, SUBMISSION, False, ref.get('status', 'unavailable'), ref.get('id'))
        else:
            flagged[0].append(i); flagged[1].append(SUBMISSION); flagged[2].append(STATUS_NONE)
        if member_1:
            ref = targets[1]
            reference(i, VALIDATION, False, ref.get('status', 'unavailable'), ref.get('id'))
    if rows:
        numeric[row_index] = rows
    if flagged[0]:
        relation_features[flagged] = 1
    if linked[0]:
        where = (linked[0], linked[1])
        relations[where], valid[where] = linked[2], True
        relation_features[(*where, slice(3, None))] = linked[3]
    encoded.update(task_numeric=numeric, task_relations=relations, task_relation_valid=valid,
        task_relation_features=relation_features, task_commands=commands,
        task_command_templates=templates, task_command_template_present=template_present)
    return encoded
