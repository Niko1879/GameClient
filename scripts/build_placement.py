"""Exact-coordinate, player-observable placement grids shared by train and play."""
from __future__ import annotations

import numpy as np

SCHEMA = 1
ANGLE_BINS = 16


def decode_grid(record, *, bins, builder, template, angle, frame):
    expected = {'schema': SCHEMA, 'bins': bins, 'builder': builder, 'template': template,
                'angle_bins': ANGLE_BINS, 'angle_index': angle, 'frame': frame}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Placement grid does not match the current build decision')
    raw = bytes.fromhex(record['mask_hex'])
    if len(raw) != (bins*bins+7)//8:
        raise ValueError('Placement grid has an invalid size')
    mask = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little', count=bins*bins)
    mask = mask.astype(bool).reshape(bins, bins)
    if int(mask.sum()) != record['legal_sites']:
        raise ValueError('Placement grid legal-site count mismatch')
    return mask


def mask_for(config, encoded, selected, template, angle):
    """The native preview checks terrain reachability and ignores hidden obstacles.

    Stored training masks are tied to a teacher prefix, never used for another
    actor/template/rotation. Live grids are queried while simulation is paused.
    """
    signature = np.asarray([selected, template, angle], dtype=np.int64)
    if 'build_signature' in encoded and np.array_equal(encoded['build_signature'], signature):
        return encoded['build_mask']
    builder = encoded['entities'][selected]['id']
    name = config['templates'][template]
    world_angle = (angle + (ANGLE_BINS//2 if encoded['rotated'] else 0)) % ANGLE_BINS
    bins = config['position_bins']
    rate = encoded['placement_source_rate']
    frame = encoded['frame']*rate//60
    candidates = [row for row in encoded.get('placement_records', [])
                  if row['builder'] == builder and row['template'] == name and row['angle_index'] == world_angle]
    if len(candidates) > 1:
        raise ValueError('Duplicate placement grids for one decision')
    if candidates:
        record = candidates[0]
    elif 'placement_query' in encoded:
        record = encoded['placement_query'](builder, name, world_angle, bins)
    else:
        raise ValueError('Missing placement grid; use a placement-capable replay export/runtime')
    mask = decode_grid(record, bins=bins, builder=builder, template=name, angle=world_angle, frame=frame)
    if encoded['rotated']:
        mask = mask[::-1, ::-1].copy()
    encoded['build_mask'], encoded['build_signature'] = mask, signature
    encoded['build_features'] = coarse_features(mask, config['map_side'])
    return mask


def coarse_features(mask, side):
    bins = len(mask)
    fine = bins//side
    return mask.reshape(side, fine, side, fine).mean(axis=(1, 3)).reshape(-1, 1)


def categorical_masks(mask, side, cell=None):
    bins, fine = len(mask), len(mask)//side
    patches = mask.reshape(side, fine, side, fine).transpose(0, 2, 1, 3).reshape(side*side, fine*fine)
    return patches.any(axis=1), None if cell is None else patches[cell]
