"""Translate supported replay command batches into the live environment's action schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from replay_training import audit, records


class ReplayActionAdapter:
    def __init__(self, catalog):
        self.catalog = {str(k): v for k, v in catalog.items()}
        self.selected = None
        self.teams = {}

    def decode(self, sample):
        actions, unsupported, rejected_builds, rejected_trains = [], [], 0, 0
        if not sample['actions']:
            # Quiet observations cannot change selection/team state. Their
            # body stays deferred until the event selector needs a DEFER.
            return dict(actions=[], unsupported=[], imitation_loss_mask=True, rejected_build_requests=0,
                        rejected_train_requests=0)
        own_rows = sample['observation']['own_entities']
        if self.selected is None:
            # Multiplayer start selects each player's first Command Center in
            # logic without a replay message (GameLogic::startNewGame,
            # findAndSelectCommandCenter). Only command samples change
            # selection, so the first one still sees that start state.
            self.selected = [e['id'] for e in own_rows if e.get('type', '').endswith('CommandCenter')][:1]
        other_rows = sample['observation'].get('known_nonowned_entities', [])
        owned = own_rows.ids() if hasattr(own_rows, 'ids') else {e['id'] for e in own_rows}
        visible = owned | (other_rows.ids(visible_only=True) if hasattr(other_rows, 'ids')
                           else {e['id'] for e in other_rows if e['visible']})
        for raw in sample['actions']:
            name = raw['name']
            values = [argument['value'] for argument in raw['arguments']]
            if 'command_result' in raw:
                # Upgrades, abilities, garrison and promotions: the engine resolved
                # the pressed button against the real selection at dispatch.
                result = raw['command_result']
                if not result['resolved']:
                    unsupported.append({'name': name, 'reason': result['reason']})
                    continue
                action = result['action']
                actors = action.get('units', [action['producer']] if 'producer' in action else [])
                if not all(i in owned for i in actors) or ('target' in action and action['target'] not in visible):
                    unsupported.append({'name': name, 'reason': 'Resolved command outside this observation'})
                    continue
                actions.append(action)
                continue
            if name == 'MSG_DOZER_CONSTRUCT':
                # Only engine-accepted construction becomes a demonstration.
                result = raw.get('build_result', {})
                if type(result.get('accepted')) is not bool:
                    raise ValueError('Missing engine construction outcome')
                if not result['accepted']:
                    rejected_builds += 1
                    continue
            try:
                if name in ('MSG_CREATE_SELECTED_GROUP', 'MSG_CREATE_SELECTED_GROUP_NO_SOUND'):
                    if values[0]:
                        self.selected = []
                    self.selected = list(dict.fromkeys(self.selected + values[1:]))
                    continue
                if name == 'MSG_DESTROY_SELECTED_GROUP':
                    self.selected = []
                    continue
                if name == 'MSG_REMOVE_FROM_SELECTED_GROUP':
                    self.selected = [i for i in self.selected if i not in values]
                    continue
                team = re.fullmatch(r'MSG_(CREATE|SELECT|ADD)_TEAM([0-9])', name)
                if team:
                    operation, index = team.groups()
                    if operation == 'CREATE':
                        # The replay message carries explicit IDs. Creating a
                        # team also removes those objects from all other teams.
                        members = [i for i in values if i in owned]
                        for other in self.teams:
                            self.teams[other] = [i for i in self.teams[other] if i not in members]
                        self.teams[index] = members
                    elif index not in self.teams:
                        raise ValueError('Unknown hotkey team')
                    elif operation == 'SELECT':
                        self.selected = self.teams[index].copy()
                    else:
                        self.selected = list(dict.fromkeys(self.selected + self.teams[index]))
                    continue
                if name in ('MSG_ENABLE_RETALIATION_MODE', 'MSG_SET_MINE_CLEARING_DETAIL'):
                    # A client option posted at game start (live play applies the same
                    # default), and the UI's preparation for a mine-clearing weapon
                    # order: settings, not decisions or waits.
                    continue
                if name == 'MSG_SET_RALLY_POINT':
                    if values[0] not in owned:
                        raise ValueError('Missing owned rally producer')
                    actions.append({'kind': 'rally', 'producer': values[0], 'position': values[1]})
                    continue
                if name == 'MSG_SELF_DESTRUCT':
                    actions.append({'kind': 'surrender'})
                    continue
                units = [i for i in self.selected if i in owned]
                if not units:
                    raise ValueError('No known owned selection')
                if name == 'MSG_RESUME_CONSTRUCTION':
                    # Native group dispatch can pick one of several builders.
                    # Preserve an unambiguous effective actor, never guess it.
                    candidates = [e['id'] for e in sample['observation']['own_entities']
                                  if e['id'] in units and values[0] in e['resume_targets']]
                    if len(candidates) != 1:
                        raise ValueError('Resume requires one unambiguous currently eligible builder')
                    actions.append({'kind': 'resume_construction', 'builder': candidates[0], 'target': values[0]})
                    continue
                verbs = {'MSG_DO_MOVETO': 'move', 'MSG_DO_ATTACKMOVETO': 'attack_move',
                         'MSG_DO_ATTACK_OBJECT': 'attack', 'MSG_DO_STOP': 'stop', 'MSG_SELL': 'sell',
                         'MSG_DOCK': 'dock'}
                if name in verbs:
                    action = {'kind': verbs[name], 'units': units.copy()}
                    if action['kind'] in ('move', 'attack_move'):
                        action['position'] = values[0]
                    elif action['kind'] in ('attack', 'dock'):
                        if values[0] not in visible:
                            raise ValueError('Target not in current owned/visible observation')
                        action['target'] = values[0]
                    actions.append(action)
                elif name in ('MSG_DOZER_CONSTRUCT', 'MSG_QUEUE_UNIT_CREATE'):
                    if len(units) != 1:
                        raise ValueError('Build/train requires a single selected entity')
                    template = self.catalog.get(str(values[0]))
                    if template is None:
                        raise ValueError('Template ID absent from replay catalog')
                    if name == 'MSG_DOZER_CONSTRUCT':
                        actions.append({'kind': 'build', 'builder': units[0], 'template': template,
                                        'position': values[1], 'angle': values[2]})
                    elif template not in next(e for e in own_rows if e['id'] == units[0])['trainable_templates']:
                        # The engine ignores a queue its own canMakeUnit/queue check
                        # refuses (full queue, no money, unit cap), which is what
                        # trainable_templates reports at dispatch: spam clicks, not
                        # demonstrations. Like a rejected build, it is UI noise.
                        rejected_trains += 1
                    else:
                        actions.append({'kind': 'train', 'producer': units[0], 'template': template})
                else:
                    raise ValueError('Command outside POC action subset')
            except (ValueError, KeyError, IndexError, TypeError) as error:
                unsupported.append({'name': name, 'reason': str(error)})
        # Selection-only samples are UI bookkeeping, not examples of strategically waiting.
        loss_mask = not unsupported and (bool(actions) or not sample['actions'])
        if len(actions) > 16 or any(len(a.get('units', [])) > 64 for a in actions):
            loss_mask = False
        return {'actions': actions, 'unsupported': unsupported, 'imitation_loss_mask': loss_mask,
                'rejected_build_requests': rejected_builds, 'rejected_train_requests': rejected_trains}


def iter_demonstrations(path, *, validate=True, player=None, validated=False):
    """Sequential observation/action/timing examples; unsupported targets are masked.

    IDs are replay-local and must be represented as pointers into each observation
    entity table, never copied as absolute IDs into another match.
    """
    if validate:
        audit(path)
        validated = True
    adapters, pending = {}, {}
    catalog = None
    for record in records(path, lazy=True, player=player, validated=validated):
        if record['record'] == 'header':
            catalog = record['template_catalog']
        elif record['record'] == 'sample':
            player = record['player_index']
            if player not in adapters:
                adapters[player] = ReplayActionAdapter(catalog)
            adapter = adapters[player]
            example = {'record': 'demonstration', 'source': 'replay',
                       'player_index': player, 'frame': record['frame'],
                       **adapter.decode(record), 'reward': None}
            # Light quiet samples (scheduled capture) carry no observation.
            if 'observation' in record:
                example['observation'] = record['observation']
            if player in pending:
                previous = pending[player]
                previous['hold_ticks'] = record['frame'] - previous['frame']
                yield previous
            pending[player] = example
        elif record['record'] == 'footer':
            for example in pending.values():
                # No subsequent decision is observed; do not invent its duration.
                example['terminal_loss_mask'] = example['imitation_loss_mask']
                example['terminal_sample'] = True
                example['hold_ticks'] = None
                example['imitation_loss_mask'] = False
                yield example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = eligible = 0
    failures = {}
    with output.open('x', encoding='utf-8') as stream:
        for example in iter_demonstrations(args.trajectory):
            stream.write(json.dumps(example) + '\n')
            total += 1
            eligible += example['imitation_loss_mask']
            for raw in example['unsupported']:
                failures[raw['name']] = failures.get(raw['name'], 0) + 1
    print(json.dumps({'examples': total, 'eligible': eligible, 'unsupported_commands': failures}, indent=2))


if __name__ == '__main__':
    main()
