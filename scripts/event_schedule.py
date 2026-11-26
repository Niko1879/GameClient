"""Clocks shared by event labels and deployed actors, and the no-op snapshot frames."""
from __future__ import annotations

import hashlib
import math

SNAPSHOT_SALT = 'noop-snapshot-v1'


def grid_ticks(config):
    """Canonical ticks between the exporter's quiet samples (event_observation_ms)."""
    return round(60*config['event_observation_ms']/1000)


def snapshot_probability(config):
    """Chance that one quiet grid frame is a no-op row: snapshot_rate per second of game time."""
    probability = config['snapshot_rate']*grid_ticks(config)/60
    if not 0 < probability <= 1:
        raise ValueError('The snapshot rate must be positive and at most one per grid frame')
    return probability


def is_snapshot(config, replay_sha256, player, frame):
    """Whether a canonical frame is a no-op row if it turns out quiet (notes/wait_snapshots.md).

    A hash of the recording, the player and the frame, so it never depends on what the player did:
    the rate is constant, and neither the last row nor the last command says what the next row is.
    """
    if frame % grid_ticks(config):
        return False
    digest = hashlib.sha256(f'{SNAPSHOT_SALT}|{replay_sha256}|{player}|{frame}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') < snapshot_probability(config)*2**64


def canonical_frame(frame, rate):
    if rate not in (30, 60) or type(frame) is not int or frame < 0:
        raise ValueError('Expected a nonnegative native frame at 30/60 Hz')
    return frame * (60 // rate)


def native_deadline(frame, rate):
    if rate not in (30, 60) or frame < 0:
        raise ValueError('Invalid event deadline')
    return math.ceil(frame * rate / 60)


def delay_mask(config, rate):
    """The delay head's legal ticks; only the memory-free (_old) policies still have one."""
    quantum = 60 // rate if rate in (30, 60) else 0
    if not quantum:
        raise ValueError('Unsupported source clock')
    return [0 < tick <= config['event_max_delay'] and tick % quantum == 0
            for tick in config['delay_ticks']]


def step_ticks(policies, frame, ordinary_ticks, rate=60):
    """Observe at the normal cadence or an earlier actor deadline."""
    if ordinary_ticks < 1:
        raise ValueError('Positive observation cadence required')
    scheduled = [p for p in policies if hasattr(p, 'next_deadline_native')]
    if not scheduled:
        return ordinary_ticks
    cadence = min(ordinary_ticks, *[max(1, math.ceil(rate*p.policy.config['event_observation_ms']/1000))
                                   for p in scheduled])
    result = cadence - frame % cadence
    for actor in scheduled:
        remaining = actor.next_deadline_native - frame
        if remaining > 0:
            result = min(result, remaining)
    return max(1, result)
