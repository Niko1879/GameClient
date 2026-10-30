"""Serializable object-family and faction identities for the learner.

Names describe the equipment's lineage, not its current owner's faction. Only
observed template names and the controlled player's own faction are consumed.
Unknown prefixes stay distinct; this is parameter sharing, not engine aliases.
"""
from __future__ import annotations

import numpy as np
import copy
from functools import lru_cache


BASE_FACTIONS = ('<unknown>', 'USA', 'China', 'GLA')
GENERALS = ('<unknown>', 'vanilla', 'air_force', 'laser', 'superweapon',
            'tank', 'infantry', 'nuke', 'toxin', 'demolition', 'stealth')
PREFIXES = {
    'AirF_': ('USA', 'air_force'), 'Lazr_': ('USA', 'laser'),
    'SupW_': ('USA', 'superweapon'), 'Tank_': ('China', 'tank'),
    'Infa_': ('China', 'infantry'), 'Nuke_': ('China', 'nuke'),
    'Chem_': ('GLA', 'toxin'), 'Demo_': ('GLA', 'demolition'),
    'Slth_': ('GLA', 'stealth'),
}
PLAYER_FACTIONS = {
    'FactionAmerica': ('USA', 'vanilla'), 'FactionChina': ('China', 'vanilla'),
    'FactionGLA': ('GLA', 'vanilla'),
    'FactionAmericaAirForceGeneral': ('USA', 'air_force'),
    'FactionAmericaLaserGeneral': ('USA', 'laser'),
    'FactionAmericaSuperWeaponGeneral': ('USA', 'superweapon'),
    'FactionChinaTankGeneral': ('China', 'tank'),
    'FactionChinaInfantryGeneral': ('China', 'infantry'),
    'FactionChinaNukeGeneral': ('China', 'nuke'),
    'FactionGLAToxinGeneral': ('GLA', 'toxin'),
    'FactionGLADemolitionGeneral': ('GLA', 'demolition'),
    'FactionGLAStealthGeneral': ('GLA', 'stealth'),
}
# Separate the major archetypes already identified by the user. Other general
# differences are represented by trainable residuals, not assumed nonexistent.
FAMILY_OVERRIDES = {
    'SupW_AmericaPatriotBattery': 'AmericaPatriotBattery::EMP',
    'Lazr_AmericaPatriotBattery': 'AmericaPatriotBattery::laser',
    'Chem_GLATunnelNetwork': 'GLATunnelNetwork::toxin',
    'Chem_GLASneakAttackTunnelNetwork': 'GLASneakAttackTunnelNetwork::toxin',
    'Chem_GLASneakAttackTunnelNetworkStart': 'GLASneakAttackTunnelNetworkStart::toxin',
}


def template_identity(name):
    """Return a candidate family plus explicit base/general identity."""
    family, base, general = name, '<unknown>', '<unknown>'
    for prefix, lineage in PREFIXES.items():
        if name.startswith(prefix) and len(name) > len(prefix):
            family, (base, general) = name[len(prefix):], lineage
            break
    else:
        for prefix, faction in (('America', 'USA'), ('China', 'China'), ('GLA', 'GLA')):
            if name.startswith(prefix):
                base, general = faction, 'vanilla'
                break
    return FAMILY_OVERRIDES.get(name, family), base, general


@lru_cache(maxsize=None)
def _vanilla_china(name):
    # Base faction and general depend only on the constant prefix table.
    return template_identity(name)[1:] == ('China', 'vanilla')


def in_production_scope(config, name):
    """Production scope is independent of ownership and observable object types."""
    scope = config.get('production_scope')
    if scope is None:
        return True
    if scope != 'vanilla_china':
        raise ValueError('Unsupported production scope: '+str(scope))
    return _vanilla_china(name)


def vanilla_china_production(config):
    """Narrow output choices while retaining complete producer/unit facts."""
    result = copy.deepcopy(config)
    result['production_scope'] = 'vanilla_china'
    allowed = {name for key, values in result['menus'].items()
               if key.rsplit('|', 1)[1] in ('build', 'train')
               and in_production_scope(result, key.rsplit('|', 1)[0])
               for name in values if in_production_scope(result, name)}
    result['templates'] = sorted(set(result['templates']) & allowed)
    if not result['templates']:
        raise ValueError('Vanilla-China production needs a nonempty native production menu')
    # Canonical command names may come from another faction's identical button,
    # so command scope follows in-scope menus and China's shortcut bar, not names.
    offered = {name for owner, names in result['command_menus'].items()
               if in_production_scope(result, owner) for name in names}
    offered.update(result['command_shortcuts'].get('FactionChina', []))
    result['commands'] = sorted(set(result['commands']) & offered)
    result['sciences'] = sorted(set(result['promotions'].get('FactionChina', [])))
    result['identity_hierarchy'] = hierarchy_config(result['types'], result['templates'])
    return result


def validate_production_scope(config):
    if config.get('production_scope') not in (None, 'vanilla_china'):
        raise ValueError('Unsupported production scope')
    if config.get('production_scope') is not None:
        if any(not in_production_scope(config, name) for name in config['templates']):
            raise ValueError('Policy template outside declared production scope')
    if any(config['command_info'].get(name, [None])[0] not in ('upgrade', 'ability') for name in config['commands']):
        raise ValueError('Policy command outside the native command catalogue')


def hierarchy_config(types, templates):
    """Freeze all mappings in the checkpoint, including unobserved catalogue types."""
    objects = ['<unknown>'] + sorted((set(types) | set(templates)) - {'<unknown>'})
    identities = [template_identity(name) for name in objects]
    families = ['<unknown>'] + sorted({row[0] for row in identities} - {'<unknown>'})
    object_ids = {name: i for i, name in enumerate(objects)}
    family_ids = {name: i for i, name in enumerate(families)}
    base_ids = {name: i for i, name in enumerate(BASE_FACTIONS)}
    general_ids = {name: i for i, name in enumerate(GENERALS)}
    return {'objects': objects, 'families': families,
            'base_factions': list(BASE_FACTIONS), 'generals': list(GENERALS),
            'object_families': [family_ids[row[0]] for row in identities],
            'object_base_factions': [base_ids[row[1]] for row in identities],
            'object_generals': [general_ids[row[2]] for row in identities],
            'type_objects': [object_ids[name] for name in types],
            'template_objects': [object_ids[name] for name in templates],
            'player_factions': {name: [base_ids[base], general_ids[general]]
                                for name, (base, general) in PLAYER_FACTIONS.items()}}


class IdentityHierarchy:
    """Validate and cache the mappings stored alongside the policy weights."""
    def __init__(self, config):
        self.config = config['identity_hierarchy']
        for key in ('objects', 'families', 'base_factions', 'generals'):
            names = self.config[key]
            if not names or names[0] != '<unknown>' or len(set(names)) != len(names):
                raise ValueError('Invalid identity hierarchy vocabulary: ' + key)
        self.objects = {name: i for i, name in enumerate(self.config['objects'])}
        count = len(self.objects)
        def indices(key, size, length):
            values = self.config[key]
            if len(values) != length or any(type(i) is not int or not 0 <= i < size for i in values):
                raise ValueError('Invalid identity hierarchy indices: ' + key)
            return np.asarray(values, dtype=int)
        self.families = indices('object_families', len(self.config['families']), count)
        self.base_factions = indices('object_base_factions', len(self.config['base_factions']), count)
        self.generals = indices('object_generals', len(self.config['generals']), count)
        self.type_objects = indices('type_objects', count, len(config['types']))
        self.template_objects = indices('template_objects', count, len(config['templates']))
        for names, mapped in ((config['types'], self.type_objects),
                              (config['templates'], self.template_objects)):
            if any(self.config['objects'][i] != name for name, i in zip(names, mapped)):
                raise ValueError('Identity hierarchy does not match exact template vocabulary')
        for lineage in self.config['player_factions'].values():
            if (len(lineage) != 2 or any(type(i) is not int for i in lineage)
                    or not 0 <= lineage[0] < len(self.config['base_factions'])
                    or not 0 <= lineage[1] < len(self.config['generals'])):
                raise ValueError('Invalid player faction hierarchy')

    def player_identity(self, faction):
        return tuple(self.config['player_factions'].get(faction, (0, 0)))
