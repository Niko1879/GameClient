"""Clocks shared by event labels and deployed actors."""
from __future__ import annotations

import math


def canonical_frame(frame, rate):
    if rate not in (30, 60) or type(frame) is not int or frame < 0:
        raise ValueError('Expected a nonnegative native frame at 30/60 Hz')
    return frame * (60 // rate)


def native_deadline(frame, rate):
    if rate not in (30, 60) or frame < 0:
        raise ValueError('Invalid event deadline')
    return math.ceil(frame * rate / 60)


def delay_mask(config, rate):
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
