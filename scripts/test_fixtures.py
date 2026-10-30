"""Small observation-v7 vanilla China fixtures shared by the policy, learner and pipeline tests."""
import copy

import numpy as np

from event_policy import default_config
from generals_policy import OBSERVATION_VERSION

VOCABULARY = {'types': ['<unknown>', 'ChinaVehicleDozer', 'ChinaCommandCenter', 'ChinaPowerPlant', 'Enemy'],
              'templates': ['ChinaPowerPlant', 'ChinaVehicleDozer'],
              'menus': {'ChinaVehicleDozer|build': ['ChinaPowerPlant'],
                        'ChinaCommandCenter|train': ['ChinaVehicleDozer']},
              # Native command catalogue shape (event_policy.command_vocabulary).
              'commands': ['Command_ArtilleryBarrage', 'Command_Capture', 'Command_UpgradeChinaRadar'],
              'command_info': {'Command_ArtilleryBarrage': ['ability', 'position'],
                               'Command_Capture': ['ability', 'object'],
                               'Command_ChinookUnload': ['evacuate', 'none'],
                               'Command_UpgradeChinaRadar': ['upgrade', 'none']},
              'command_menus': {'ChinaCommandCenter': ['Command_UpgradeChinaRadar', 'Command_ArtilleryBarrage'],
                                'ChinaVehicleDozer': ['Command_Capture']},
              'command_shortcuts': {'FactionChina': ['Command_ArtilleryBarrage']},
              'promotions': {'FactionChina': ['SCIENCE_ArtilleryBarrage1', 'SCIENCE_Frenzy1']},
              'upgrades': ['Upgrade_ChinaRadar', 'Upgrade_InfantryCaptureBuilding']}
SMALL = dict(hidden=16, entity_hidden=8, embedding=4, map_hidden=6, scalar_hidden=6, map_side=8, position_bins=16)


def task(kind='idle', **changes):
    return {'available': True, 'kind': kind, 'destination': None,
            'target': {'status': 'none'}, 'combat_target': {'status': 'none'},
            'cargo': {'available': False}, **changes}


def entity(identifier, name, own, position, visible=True, **changes):
    row = {'id': identifier, 'type': name, 'own': own, 'owner': 2 if own else 3, 'position': position,
           'health': 100, 'max_health': 100, 'visible': visible, 'last_seen_frame': 1}
    if own:
        mobile = name != 'ChinaCommandCenter'
        row.update(construction_percent=-1, production_queue=[], task=task(), radius=10,
                   has_weapon=mobile, can_attack=mobile, has_atk_move_cmd=mobile, can_attack_move=mobile,
                   dock_targets=[], resume_targets=[],
                   buildable_templates=['ChinaPowerPlant'] if name == 'ChinaVehicleDozer' else [],
                   trainable_templates=['ChinaVehicleDozer'] if name == 'ChinaCommandCenter' else [],
                   commands=[], command_cooldowns=[], command_targets=[], enter_targets=[],
                   container_id=0, occupants=0, capacity=0, upgrades=[], can_guard=mobile,
                   repair_targets=[], repair_depots=[])
    row.update(changes)
    return row


def observation_fixture():
    terrain = {'grid_size': 32, 'extent_low': [0, 0, 0], 'extent_high': [1000, 1000, 100],
               'height': [0] * 1024, 'cliff': [0] * 1024}
    observation = {'player': {'index': 2, 'faction': 'FactionChina', 'cash': 10000,
                              'power_production': 0, 'power_consumption': 0, 'radar': False,
                              'rank': 1, 'skill_points': 0, 'science_points': 1, 'dead': False,
                              'sciences': [], 'purchasable_sciences': [], 'upgrades': []},
                   'own_entities': [entity(10, 'ChinaVehicleDozer', True, [100, 100, 0],
                                           task=task('move', moving=True, destination=[300, 400, 0])),
                                    entity(20, 'ChinaCommandCenter', True, [120, 120, 0], builder_id=10)],
                   'known_nonowned_entities': [entity(99, 'Enemy', False, [800, 800, 0]),
                                               entity(100, 'Enemy', False, [850, 850, 0], False)],
                   'visibility_grid': [2] * 1024, 'observation_policy_version': OBSERVATION_VERSION,
                   'last_submission': {'available': True, 'kind': 'move', 'actors': [10],
                                       'position': [300, 400, 0], 'result': 'submitted', 'age_seconds': .1},
                   'last_validation': {'available': False}}
    return copy.deepcopy(VOCABULARY), observation, terrain


def fixture(placement_mode='fast'):
    vocabulary, observation, terrain = observation_fixture()
    config = default_config(vocabulary, placement_mode)
    config.update(SMALL)
    return config, observation, terrain


def raw_row(frame, observation, actions=(), *, ui=False, unsupported=False, terminal=False):
    return {'frame': frame, 'observation': copy.deepcopy(observation), 'actions': list(actions),
            'unsupported': [{'name': 'unavailable', 'reason': 'test'}] if unsupported else [],
            'imitation_loss_mask': not (ui or unsupported), 'hold_ticks': None if terminal else 6,
            'terminal_sample': terminal}


def grid(mask, *, angle=0, frame=0, builder=10, template='ChinaPowerPlant'):
    """A native placement preview record for one builder, template and rotation."""
    return {'schema': 1, 'bins': len(mask), 'angle_bins': 16, 'angle_index': angle,
            'builder': builder, 'template': template, 'frame': frame,
            'mask_hex': np.packbits(mask.reshape(-1), bitorder='little').tobytes().hex(),
            'legal_sites': int(mask.sum())}
