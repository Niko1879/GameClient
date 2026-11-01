"""Causal action/DEFER selection before neural feature computation.

Dense exports remain authoritative. UI messages update ReplayActionAdapter but
do not reset decision or command clocks. Unsupported orders censor delay labels.
Out-of-scope production ends only that player's training episode.
"""
from __future__ import annotations

from itertools import chain

from generals_policy import KINDS, Unrepresentable, initial_home
from generals_identity import in_production_scope


def outside_production_scope(config, row, action):
    if config.get('production_scope') is None or action['kind'] not in ('build', 'train'):
        return False
    template = action.get('template')
    if not isinstance(template, str):
        return False
    if not in_production_scope(config, template):
        return True
    actor = action.get('builder', action.get('producer'))
    entity = next((e for e in row.get('observation', {}).get('own_entities', ()) if e['id'] == actor), None)
    # Missing actors retain the existing structural-rejection path.
    return entity is not None and not in_production_scope(config, entity['type'])


def decisions(rows, config, counts, validate=None, observe=None):
    """Selected candidates in source order.

    `observe(last_decision, finished)` runs after every consumed row: the
    scheduled-capture controller derives which quiet observations can still
    be selected from this state alone (see scheduled_capture.py).
    """
    cap = config['event_max_delay']
    pending, quiet, last_decision = None, None, None
    previous_source_frame = None

    def candidate(row, actions):
        return {'source': row, 'actions': actions, 'previous_frame': last_decision}

    def close(following=None):
        nonlocal pending
        if pending is not None:
            pending['delay_ticks'] = None if following is None else following-pending['source']['frame']
            if pending['delay_ticks'] is None:
                counts['censored_delays'] += 1
            result, pending = pending, None
            return result

    def checked(event):
        return validate(event) if validate is not None else True

    for row in rows:
        counts['source_observations'] += 1
        frame, actions = row['frame'], row['actions']
        counts['rejected_build_requests'] += row.get('rejected_build_requests', 0)
        if previous_source_frame is not None:
            if frame <= previous_source_frame:
                raise ValueError('Event observations must advance in source time')
            counts['observed_canonical_ticks'] += frame-previous_source_frame
        previous_source_frame = frame
        terminal = row.get('terminal_sample', False)
        eligible = row.get('terminal_loss_mask', row['imitation_loss_mask'])
        gameplay = bool(actions or row['unsupported'])
        counts['gameplay_batches'] += gameplay
        for unsupported in row['unsupported']:
            counts['unsupported:'+unsupported['name']] += 1
        ui_only = not gameplay and not eligible and not terminal
        counts['raw_commands'] += len(actions)
        for action in actions:
            counts['raw_kind:'+action['kind']] += 1
        if ui_only:
            counts['ui_only_messages'] += 1
        foreign = [action for action in actions if outside_production_scope(config, row, action)] if actions else ()
        if foreign:
            result = close()
            if result is not None:
                yield result
            counts['excluded:production_scope'] += 1
            for action in foreign:
                counts['excluded:production_scope:'+action['kind']] += 1
            counts['truncated:production_scope'] += 1
            # write_shard records this per player, rather than summing frames.
            counts['production_scope_cutoff_frame'] = frame
            if observe is not None:
                observe(last_decision, True)
            break
        if last_decision is not None and frame > last_decision+cap:
            if quiet is not None and quiet['source']['frame'] > last_decision:
                if not checked(quiet):
                    raise ValueError('DEFER must remain representable')
                result = close(quiet['source']['frame'])
                if result is not None:
                    yield result
                quiet['previous_frame'] = last_decision
                pending, last_decision, quiet = quiet, quiet['source']['frame'], None
            if frame > last_decision+cap:
                result = close()
                if result is not None:
                    yield result
                counts['observation_gaps'] += 1
                last_decision = None
        if gameplay:
            reason = None
            if row['unsupported'] or not eligible:
                reason = 'unsupported'
            elif len(actions) != 1 or actions[0]['kind'] not in KINDS[1:]:
                reason = 'multi_or_unavailable'
            proposed = None
            if reason is None:
                proposed = candidate(row, actions)
                if not checked(proposed):
                    reason = 'structural'
            if reason:
                result = close()
                if result is not None:
                    yield result
                counts['excluded:'+reason] += 1
                for action in actions:
                    counts['excluded:'+reason+':'+action['kind']] += 1
                quiet = None
            else:
                result = close(frame)
                if result is not None:
                    yield result
                pending = proposed
                last_decision, quiet = frame, None
        elif not ui_only and not terminal:
            if last_decision is None or frame >= last_decision+cap:
                proposed = candidate(row, [])
                if not checked(proposed):
                    raise ValueError('DEFER must remain representable')
                result = close(frame)
                if result is not None:
                    yield result
                pending = proposed
                last_decision, quiet = frame, None
            else:
                quiet = candidate(row, [])
        if terminal:
            result = close()
            if result is not None:
                yield result
            if observe is not None:
                observe(last_decision, True)
            break
        if observe is not None:
            observe(last_decision, False)
    result = close()
    if result is not None:
        yield result


class EventValidator:
    """Encode a selected candidate; reject it if its target is unrepresentable.

    With schedule_only (the scheduled-capture controller) a selected wait is
    accepted without encoding: decisions() requires every wait to be
    representable, so only command acceptance shapes the schedule. A command
    checks only what policy.target() reads, and nothing is kept for encoding.
    """
    def __init__(self, policy, game, counts, first_observation, *, schedule_only=False):
        self.policy, self.game, self.counts, self.schedule_only = policy, game, counts, schedule_only
        self.home = initial_home(first_observation)

    def __call__(self, event):
        source = event['source']
        if 'observation' not in source:
            raise ValueError('Scheduled capture omitted a selected observation')
        if self.schedule_only and not event['actions'] and self.home is not None:
            return True
        # Detach selected binary data before encoding/buffering so chunk memory
        # accounting sees ordinary containers and retains no source block.
        from binary_replay import materialize
        observation = materialize(source['observation'])
        if self.home is None:
            self.home = initial_home(observation)
        if self.schedule_only and not event['actions']:
            return True
        game, policy = self.game, self.policy
        if self.schedule_only:
            encoded = policy.encoder.schedule_inputs(observation, game['map'])
        else:
            encoded = policy.encoder.encode(observation, game['map'], source['frame'],
                previous_frame=event['previous_frame'], home_position=self.home,
                source_ticks_per_second=game['source_ticks_per_second'])
            encoded['event_rate'] = game['source_ticks_per_second']
        try:
            target = policy.target(encoded, event['actions'])
        except (Unrepresentable, KeyError) as error:
            self.counts['structural_exclusions'] += 1
            self.counts['structural:'+str(error)] += 1
            return False
        if self.schedule_only:
            return True
        event['prepared'] = {'encoded': encoded, 'target': target, 'hold_ticks': 0,
               'scope': 'event', 'frame': source['frame'], 'player': game['player'],
               'source_ticks_per_second': game['source_ticks_per_second']}
        return True


def encoded_events(policy, game, counts):
    """Encode selected candidates before committing their causal actor history."""
    iterator = iter(game['rows'])
    first = next(iterator, None)
    if first is None:
        return
    validate = EventValidator(policy, game, counts, first['observation'])
    for event in decisions(chain([first], iterator), policy.config, counts, validate=validate):
        row = event['prepared']
        target = row['target']
        duration = event['delay_ticks']
        if duration is not None:
            if duration not in policy.config['delay_ticks']:
                raise ValueError('Selected event exceeds the scheduling cap')
            target['delay'] = policy.config['delay_ticks'].index(duration)
            row['hold_ticks'] = duration
        counts['decisions'] += 1
        counts['defers' if target['kind'] == 'wait' else 'commands'] += 1
        counts['retained_kind:'+target['kind']] += 1
        yield row
