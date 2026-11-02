"""Check recorded live rollouts and summarize policy command coverage."""
import argparse
from collections import Counter
import json

from replay_training import finite, records, require


def placement_candidates(settings):
    """Size of the native snap disc searched per build order (SelfPlay.cpp validateAction)."""
    radius = settings.get('build_snap_radius', 0.0)
    if radius <= 0:
        return 1
    step = settings['build_snap_step']
    extent = int(radius // step)
    return sum(1 for y in range(-extent, extent+1) for x in range(-extent, extent+1)
               if (x*x + y*y)*step*step <= radius*radius)


def audit_rollout(path):
    header = footer = None
    previous_frame = None
    steps = 0
    accepted, rejected = Counter(), Counter()
    final_types = {}
    final_transition = None
    for record in records(path):
        finite(record)
        require(footer is None, 'Record after rollout footer')
        kind = record['record']
        if kind == 'header':
            require(header is None and steps == 0, 'Duplicate rollout header')
            require(record['source'] == 'self_play', 'Unsupported rollout source')
            header = record
            placement_limit = placement_candidates(record['execution_settings'])
            controlled = record['controlled_players']
            require(controlled and len(controlled) == len(set(controlled)) and set(controlled) <= set(record['players']),
                    'Invalid controlled player list')
        elif kind == 'transition':
            require(header is not None, 'Transition without header')
            require(record['next_frame'] > record['frame'], 'Nonpositive transition duration')
            require(previous_frame is None or previous_frame == record['frame'], 'Discontinuous rollout')
            previous_frame = record['next_frame']
            require(not (record['terminated'] and record['truncated']), 'Conflicting episode flags')
            require(set(record['actions']) <= set(map(str, header['controlled_players'])),
                    'Action batch includes a built-in AI player')
            for observations in (record['observations'], record['next_observations']):
                require(set(observations) == set(map(str, header['players'])), 'Wrong observation players')
                for key, observation in observations.items():
                    require(observation['player']['index'] == int(key), 'Player identity mismatch')
                    require(all(e['own'] and e['owner'] == int(key) for e in observation['own_entities']),
                            'Foreign owned entity')
                    require(all(not any(field in e for field in ('production_queue', 'ai_state', 'construction_percent',
                                                                 'buildable_templates', 'trainable_templates', 'builder_id', 'producer_id',
                                                                 'has_weapon', 'can_attack', 'has_atk_move_cmd', 'can_attack_move', 'dock_targets',
                                                                 'resume_targets', 'commands', 'command_cooldowns', 'command_targets',
                                                                 'enter_targets', 'container_id', 'occupants', 'capacity', 'upgrades',
                                                                 'can_guard', 'repair_targets', 'repair_depots',
                                                                 'weapon_cooldown_ticks', 'ticks_since_damage'))
                                for e in observation['known_nonowned_entities']), 'Private enemy state leaked')
            for feedback in record['action_feedback']:
                require(feedback['player'] in header['controlled_players'], 'Command issued to a built-in AI player')
                if 'placement_attempts' in feedback:
                    require(feedback['action']['kind'] == 'build' and type(feedback['placement_attempts']) is int
                            and 1 <= feedback['placement_attempts'] <= placement_limit, 'Invalid placement retry count')
                if 'executed_action' in feedback:
                    original, executed = feedback['action'], feedback['executed_action']
                    require(feedback['accepted'] and original['kind'] == 'build', 'Unexpected resolved action')
                    require({k:v for k,v in original.items() if k != 'position'} ==
                            {k:v for k,v in executed.items() if k != 'position'},
                            'Placement resolver changed nonposition arguments')
                    require(isinstance(executed.get('position'), list) and len(executed['position']) == 3,
                            'Invalid executed placement position')
                if feedback['accepted']:
                    accepted[feedback['action']['kind']] += 1
                else:
                    rejected[feedback['reason']] += 1
            final_types = {p: dict(Counter(e['type'] for e in o['own_entities']))
                           for p, o in record['next_observations'].items()}
            if record['terminated']:
                require(sorted(record['rewards'].values()) in ([-1, 1], [0, 0]), 'Wrong terminal rewards')
            else:
                require(all(r == 0 for r in record['rewards'].values()), 'Nonterminal reward')
            dead = {p: bool(o['player']['dead']) for p, o in record['next_observations'].items()}
            require(set(record['rewards']) == set(dead), 'Wrong reward players')
            require(record['terminated'] == any(dead.values()), 'Terminal flag disagrees with player defeat state')
            expected = {p: (-1 if lost else 1) if any(dead.values()) and not all(dead.values()) else 0
                        for p, lost in dead.items()}
            require(record['rewards'] == expected, 'Reward disagrees with player defeat state')
            final_transition = record
            steps += 1
        elif kind == 'footer':
            footer = record
        else:
            raise ValueError(f'Unexpected rollout record {kind}')
    require(header is not None and footer is not None and footer['completed'], 'Incomplete rollout')
    require(footer['steps'] == steps and footer['frame'] == previous_frame, 'Rollout footer mismatch')
    require(footer['terminated'] or footer['truncated'], 'Episode did not end')
    require(final_transition is not None, 'Rollout has no transitions')
    for key in ('rewards', 'terminated', 'truncated'):
        require(footer[key] == final_transition[key], 'Footer disagrees with final transition: ' + key)
    return {'steps': steps, 'final_frame': previous_frame, 'terminated': footer['terminated'],
            'truncated': footer['truncated'], 'accepted_actions': dict(accepted),
            'rejected_actions': dict(rejected), 'final_owned_types': final_types,
            'rewards': footer['rewards'], 'controlled_players': header['controlled_players']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory')
    args = parser.parse_args()
    print(json.dumps(audit_rollout(args.trajectory), indent=2))
