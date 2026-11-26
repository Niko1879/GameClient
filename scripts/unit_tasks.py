"""Owned units' current tasks and what they are attacking, from the observation alone (memory-free).

Feature review (user, 26 Nov 2026): no receipt of the player's last order, no membership of it, no builder,
producer or preferred-dock links; the physical flags velocity, firing and the target already show are gone.
"""
from __future__ import annotations

import math
import numpy as np

TASKS = ('unknown', 'idle', 'move', 'attack_move', 'attack', 'construct',
         'collect', 'return_supplies', 'dock', 'produce', 'guard', 'repair', 'ability', 'other')
# What the unit is attacking: its ordered target, and the victim it is engaging now.
RELATIONS = ('target', 'combat_target')
STATUS = ('none', 'current', 'unavailable')
# Task one-hot, available, disabled; destination present, its point, the offset to it and its squared length;
# supply interface, cargo fill, ferrying.
NUMERIC = len(TASKS) + 2 + 6 + 3
RELATION_FEATURES = len(STATUS) + 3
RELATION_INDEX = {name: i for i, name in enumerate(RELATIONS)}
STATUS_INDEX = {name: i for i, name in enumerate(STATUS)}


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid unit-task '+name)
    return value


def _one_hot(vocabulary):
    return {name: [float(name == other) for other in vocabulary] for name in vocabulary}


TASK_ONE_HOT = _one_hot(TASKS)
STATUS_NONE = STATUS_INDEX['none']
NO_TARGET = {'status': 'none'}


def encode(config, observation, encoded, *, rotated):
    entities = encoded['entities']
    count = len(encoded['numeric'])
    numeric = np.zeros((count, NUMERIC))
    relations = np.zeros((count, len(RELATIONS)), dtype=np.int64)
    valid = np.zeros_like(relations, dtype=bool)
    relation_features = np.zeros((count, len(RELATIONS), RELATION_FEATURES))
    (low_x, low_y), (span_x, span_y) = encoded['low'].tolist(), encoded['span'].tolist()

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

    def reference(row, index, status, identity):
        if status not in STATUS_INDEX:
            raise ValueError('Unknown unit-task reference status')
        target = ids.get(identity) if status == 'current' else None
        if target is not None:
            other = entities[target]
            if not (other['own'] or other['visible']):
                target = None
        if status == 'current' and target is None:
            status = 'unavailable'
        flagged[0].append(row); flagged[1].append(index); flagged[2].append(STATUS_INDEX[status])
        if target is not None:
            mine, theirs = positions[row], positions[target]
            dx, dy = theirs[0]-mine[0], theirs[1]-mine[1]
            linked[0].append(row); linked[1].append(index); linked[2].append(target)
            linked[3].append((dx, dy, dx*dx+dy*dy))

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
        # The engine exposes SupplyTruckAIInterface on units such as Helixes
        # that cannot carry supplies. A zero-capacity interface is not cargo.
        rows.append([*one_hot, 1, float(bool(task.get('disabled', False))),
                     float(destination is not None), point_x, point_y, delta_x, delta_y, delta_x*delta_x+delta_y*delta_y,
                     float(supply_interface and capacity > 0), boxes/max(capacity, 1), float(bool(cargo.get('ferrying', False)))])
        row_index.append(i)
        for index, role in enumerate(RELATIONS):
            ref = task.get(role, NO_TARGET)
            status = ref.get('status', 'unavailable')
            if status == 'none':
                flagged[0].append(i); flagged[1].append(index); flagged[2].append(STATUS_NONE)
            else:
                reference(i, index, status, ref.get('id'))
    if rows:
        numeric[row_index] = rows
    if flagged[0]:
        relation_features[flagged] = 1
    if linked[0]:
        where = (linked[0], linked[1])
        relations[where], valid[where] = linked[2], True
        relation_features[(*where, slice(3, None))] = linked[3]
    encoded.update(task_numeric=numeric, task_relations=relations, task_relation_valid=valid,
                   task_relation_features=relation_features)
    return encoded
