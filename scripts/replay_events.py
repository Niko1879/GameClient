"""Causal command row selection before neural feature computation.

Dense exports remain authoritative. A row is a command the policy can represent; several
commands on one tick (lag bunching) become one row each, in source order, sharing that
tick's observation. There are no no-op rows: pace trains on the time to the next order
(event_dataset.delay_targets). UI messages update ReplayActionAdapter but are never rows.
Out-of-scope production ends only that player's training episode.
"""
from __future__ import annotations

from itertools import chain

from generals_policy import KINDS, Unrepresentable, initial_home
from generals_identity import in_production_scope, production_twin


def outside_production_scope(config, row, action):
    if config.get('production_scope') is None or action['kind'] not in ('build', 'train'):
        return False
    template = action.get('template')
    if not isinstance(template, str):
        return False
    if not in_production_scope(config, production_twin(config, template)):
        return True
    actor = action.get('builder', action.get('producer'))
    entity = next((e for e in row.get('observation', {}).get('own_entities', ()) if e['id'] == actor), None)
    # Missing actors retain the existing structural-rejection path.
    return entity is not None and not in_production_scope(config, production_twin(config, entity['type']))


def decisions(rows, config, counts, validate=None):
    """Selected candidates in source order: one per representable command."""
    previous_source_frame = None

    def candidate(row, actions):
        return {'source': row, 'actions': actions}

    def exclude(reason, actions):
        counts['excluded:'+reason] += 1
        for action in actions:
            counts['excluded:'+reason+':'+action['kind']] += 1

    def checked(event):
        return validate(event) if validate is not None else True

    for row in rows:
        counts['source_observations'] += 1
        frame, actions = row['frame'], row['actions']
        counts['rejected_build_requests'] += row.get('rejected_build_requests', 0)
        counts['rejected_train_requests'] += row.get('rejected_train_requests', 0)
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
            counts['excluded:production_scope'] += 1
            for action in foreign:
                counts['excluded:production_scope:'+action['kind']] += 1
            counts['truncated:production_scope'] += 1
            # write_shard records this per player, rather than summing frames.
            counts['production_scope_cutoff_frame'] = frame
            return
        if gameplay:
            if row['unsupported'] or not eligible:
                exclude('unsupported', actions)
            else:
                counts['split_ticks'] += len(actions) > 1
                for action in actions:
                    if action['kind'] not in KINDS[1:]:
                        exclude('unavailable', [action])
                        continue
                    proposed = candidate(row, [action])
                    if checked(proposed):
                        yield proposed
                    else:
                        exclude('structural', [action])
        if terminal:
            return


class EventValidator:
    """Encode a selected candidate; reject it if its target is unrepresentable."""
    def __init__(self, policy, game, counts, first_observation):
        self.policy, self.game, self.counts = policy, game, counts
        self.home = initial_home(first_observation)

    def __call__(self, event):
        source = event['source']
        if 'observation' not in source:
            raise ValueError('Scheduled capture omitted a selected observation')
        # Detach selected binary data before encoding/buffering so chunk memory
        # accounting sees ordinary containers and retains no source block.
        from binary_replay import materialize
        observation = materialize(source['observation'])
        if self.home is None:
            self.home = initial_home(observation)
        game, policy = self.game, self.policy
        encoded = policy.encoder.encode(observation, game['map'], source['frame'], home_position=self.home,
            source_ticks_per_second=game['source_ticks_per_second'])
        encoded['event_rate'] = game['source_ticks_per_second']
        try:
            target = policy.target(encoded, event['actions'])
        except (Unrepresentable, KeyError) as error:
            self.counts['structural_exclusions'] += 1
            self.counts['structural:'+str(error)] += 1
            return False
        event['prepared'] = {'encoded': encoded, 'target': target,
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
        counts['decisions'] += 1
        counts['defers' if target['kind'] == 'wait' else 'commands'] += 1
        counts['retained_kind:'+target['kind']] += 1
        yield row
