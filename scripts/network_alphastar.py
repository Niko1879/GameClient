"""TEMPORARY comparison network (notes/README.md PLAN, 24 Nov): AlphaGenerals with the 70M's core and heads.

The AlphaGenerals encoder and argument heads, with three of the 70M's (v11, AlphaStar-style) choices:
- a memory-free core: two tanh layers 3840 wide (the 70M's event_input/event_hidden), no LSTM;
- AlphaStar's action-type head: 16 pre-activation residual blocks before the flat action logits
  (the 70M's verb head without its scalar-context gate, which AlphaGenerals' scalars already feed);
- skip connections from the map ResNet into the location head's FiLM blocks (each projected to SKIP
  channels: full-width skips for every row of a 15,360-row update would not fit the GPU).

Selected by the config entry network='alphastar' (model-only: not in event_store.DATA_CONFIG, so no feature key
includes it).
network='memoryfree' takes only the first change (the memory-free core; AlphaGenerals' flat action layer, no skips),
to tell how much of any difference is memory. network='paced' is memory-free with no trained no-op (orders only)
and a pace head on the detached core, trained on the delay to the player's next order, that sets the live order
rate (notes/results.md, 26 Nov). network='movement' is the memory-free core with a location head that
sees the selected units and every entity relative to them (notes/movement.md). Config effects='map' (EFFECTS_MAP; any
network but 'alphastar') drops the effect tokens from the transformer: effects are painted on the map beside the units,
and every unit then reads the map at its own cell (notes/movement.md, "Effects on the map").
It plays only through TorchLivePolicy (no NumPy equations). When the comparison is decided,
delete this file and the lines that name it (grep network_alphastar / 'alphastar'): one pipeline in the end.
"""
from __future__ import annotations

import math

import numpy as np

import torch
from torch.nn import functional as F

from generals_policy import LOCATION_BLOCKS, LOGIT_CAP, MAP_POOL, MAP_RESIDUAL_BLOCKS, MAP_STAGES, SCATTER_SCALE, \
    SCATTER_VOLUME, UNDER_ATTACK_COLUMN, location_stages
from torch_policy_v3 import TorchPolicyV3

NETWORK = 'alphastar'
# Memory-free, pace (from the delay label) instead of a trained no-op.
PACED = 'paced'
# Memory-free core, location head with the selected-units plane, relevance planes and whole-map attention.
MOVEMENT = 'movement'
# Paced with a deep, narrower core (user, 27-28 Nov; pilot in notes/results.md): CORE_BLOCKS pre-norm residual blocks
# of width DEEP_CORE (inner x CORE_EXPANSION) where SCC has its LSTM; single-layer heads as memory-free.
DEEP = 'deep'
NETWORKS = ('alphastar', 'memoryfree', PACED, MOVEMENT, DEEP)
CORE_BLOCKS, DEEP_CORE, CORE_EXPANSION = 12, 768, 2
# Each block's output layer starts at .1 x sqrt(PILOT_BLOCKS/CORE_BLOCKS): the summed residual branches start as large
# as the 6-block pilot's. No weight decay (28 Nov): train and validation loss agree at this scale, nothing to curb.
PILOT_BLOCKS = 6
# Config effects='map': effects painted on the map, units read their cell; no effect tokens.
EFFECTS_MAP = 'map'
# The 70M's core width (its 'hidden'); the heads read it as AlphaGenerals' heads read the LSTM output.
WIDTHS = {'core': 3840}


def variant_widths(config):
    """Widths a comparison network sets over generals_policy.WIDTHS: the wide core, or the deep network's narrow one."""
    return {'core': DEEP_CORE} if config.get('network') == DEEP else WIDTHS
ACTION_BLOCKS, ACTION_WIDTH = 16, 256
SKIP = 32
# Pace: the hidden width of its own small network.
PACE_HIDDEN = 256
# Movement: relevance heads and their width, whole-map attention heads.
RELEVANCE_HEADS, RELEVANCE_WIDTH, CELL_HEADS = 8, 32, 4


def add_parameters(w, b, c, W, config, actions):
    """The core (and for 'alphastar' the action head and skips) in place of the LSTM and the linear action layer."""
    core, m = W['core'], config['map_hidden']
    if config['network'] != DEEP:
        w('core_hidden_w', (core, core)); b('core_hidden_b', core)
    if config.get('effects') == EFFECTS_MAP:
        if config['network'] == NETWORK:
            raise ValueError("effects='map' needs TorchPolicyV3's map encoder; 'alphastar' has its own")
        d, scatter = W['entity'], W['scatter']
        # Each effect token (effect_w, effect_type_w, effect_power_w) to the scatter width, then into the map
        # input beside the unit image (map_project's outputs).
        w('effect_scatter_w', (d, scatter), .5); b('effect_scatter_b', scatter)
        w('effect_map_w', (scatter, scatter))
        # Units read the 1x1 map at their cell into their features and keys; small start.
        w('unit_cell_w', (W['map_out'], d), .1); b('unit_cell_b', d)
        w('unit_cell_key_w', (W['map_out'], W['key']), .1)
    if config['network'] in (PACED, DEEP):
        # log(order rate here / the player's own game mean): an output only, never read by another head, on a
        # detached core (no pace gradient reaches the trunk). Zero start: the player's mean pace everywhere.
        w('pace_hidden_w', (core, PACE_HIDDEN), math.sqrt(2)); b('pace_hidden_b', PACE_HIDDEN)
        w('pace_w', (PACE_HIDDEN, 1), 0); b('pace_b', 1)
    if config['network'] == DEEP:
        # Pre-norm residual blocks after core_in (FC-ReLU): x + B relu(A norm(x)); B starts small.
        for block in range(CORE_BLOCKS):
            b(f'core_res{block}_norm_g', core, 1.); b(f'core_res{block}_norm_b', core)
            w(f'core_res{block}_a', (core, CORE_EXPANSION*core), math.sqrt(2)); b(f'core_res{block}_a_b', CORE_EXPANSION*core)
            w(f'core_res{block}_b', (CORE_EXPANSION*core, core), .1*math.sqrt(PILOT_BLOCKS/CORE_BLOCKS))
            b(f'core_res{block}_b_b', core)
        return
    if config['network'] == PACED:
        return
    if config['network'] == MOVEMENT:
        key, cells = W['key'], config['map_side']**2
        h = RELEVANCE_HEADS*RELEVANCE_WIDTH
        # Relevance: the location query (4 planes of n x n) asks every entity's key, per head.
        w('relevance_q_w', (4*cells, h)); b('relevance_q_b', h)
        w('relevance_k_w', (key, h))
        # The selected-units plane and the relevance planes enter beside location_input.
        c('location_extra', m, 1+RELEVANCE_HEADS, 1)
        # One pre-norm attention block over the n x n cells, then a small MLP; outputs start small.
        w('cell_position', (cells, m), .1)
        b('cell_attention_norm_g', m, 1.); b('cell_attention_norm_b', m)
        w('cell_qkv_w', (m, 3*m)); b('cell_qkv_b', 3*m)
        w('cell_out_w', (m, m), .1); b('cell_out_b', m)
        b('cell_mlp_norm_g', m, 1.); b('cell_mlp_norm_b', m)
        w('cell_mlp_a_w', (m, 2*m), math.sqrt(2)); b('cell_mlp_a_b', 2*m)
        w('cell_mlp_b_w', (2*m, m), .1); b('cell_mlp_b_b', m)
        return
    if config['network'] != NETWORK:
        return
    action_head(w, b, core, actions)
    # Skips: the map before its residual blocks and after each but the last, one per FiLM block.
    for skip in range(LOCATION_BLOCKS):
        c(f'map_skip{skip}', SKIP, m, 1)
        c(f'location_skip{skip}', m, SKIP, 1)
    if LOCATION_BLOCKS != MAP_RESIDUAL_BLOCKS:
        raise ValueError('One map skip per location block')


def action_head(w, b, core, actions):
    """AlphaStar's action-type head: ACTION_BLOCKS pre-activation residual blocks of ACTION_WIDTH, then the logits."""
    w('verb_in_w', (core, ACTION_WIDTH), math.sqrt(2)); b('verb_in_b', ACTION_WIDTH)
    for block in range(ACTION_BLOCKS):
        # The branch ends in a linear layer started small, so a block can add or subtract from the stream.
        w(f'verb_res{block}_a', (ACTION_WIDTH, ACTION_WIDTH), math.sqrt(2)); b(f'verb_res{block}_a_b', ACTION_WIDTH)
        w(f'verb_res{block}_b', (ACTION_WIDTH, ACTION_WIDTH), .1); b(f'verb_res{block}_b_b', ACTION_WIDTH)
    w('action_w', (ACTION_WIDTH, actions), .1); b('action_b', actions)


class AlphaStarTorchPolicy(TorchPolicyV3):
    """TorchPolicyV3 with the 70M's core, action head and location skips; 'map' holds [1x1 map ‖ skips].
    For 'memoryfree' only the core: the head, map and location are TorchPolicyV3's."""

    def initial_state(self, lanes):
        # A one-wide placeholder keeps the lane-state plumbing of the LSTM network; the core never reads it.
        zeros = torch.zeros((lanes, 1), dtype=self.dtype, device=self.device)
        return [(zeros, zeros)]

    def core(self, core_input, resets, state):
        p = self.weights
        with torch.autocast(self.device.type, enabled=False):
            if self.config['network'] == DEEP:
                # FC-ReLU, then pre-norm residual blocks, then a parameter-free norm for the heads.
                x = F.relu(core_input.to(self.dtype) @ p['core_in_w']+p['core_in_b'])
                for block in range(CORE_BLOCKS):
                    h = F.layer_norm(x, (x.shape[-1],), p[f'core_res{block}_norm_g'], p[f'core_res{block}_norm_b'], eps=1e-5)
                    x = x+F.relu(h @ p[f'core_res{block}_a']+p[f'core_res{block}_a_b']) @ p[f'core_res{block}_b']+p[f'core_res{block}_b_b']
                return self.normalize(x), state
            hidden = torch.tanh(core_input.to(self.dtype) @ p['core_in_w']+p['core_in_b'])
            return torch.tanh(hidden @ p['core_hidden_w']+p['core_hidden_b']), state

    def pace(self, core):
        """Predicted log(order rate here / the player's own game mean), per row, trained on the delay to the next
        order (event_dataset.delay_targets); segregated: it reads a detached core, so it never shapes it."""
        p = self.weights
        hidden = F.relu(core.detach() @ p['pace_hidden_w']+p['pace_hidden_b'])
        return (hidden @ p['pace_w']+p['pace_b'])[..., 0]

    def action_logits(self, core):
        if self.config['network'] != NETWORK:
            return super().action_logits(core)
        p = self.weights
        x = F.relu(core @ p['verb_in_w']+p['verb_in_b'])
        for block in range(ACTION_BLOCKS):
            y = F.relu(self.normalize(x)) @ p[f'verb_res{block}_a']+p[f'verb_res{block}_a_b']
            x = x+F.relu(self.normalize(y)) @ p[f'verb_res{block}_b']+p[f'verb_res{block}_b_b']
        return F.relu(x) @ p['action_w']+p['action_b']

    def effect_image(self, inputs, tokens):
        """For effects='map': each effect token painted at its pixel, as encode_map paints units (shared per
        pixel beyond SCATTER_VOLUME), projected into the map input: (batch, scatter, side, side)."""
        if self.config.get('effects') != EFFECTS_MAP:
            return None
        p, config = self.weights, self.config
        side = config['map_side']*SCATTER_SCALE//MAP_POOL
        pixels, placed = inputs['effect_pixels'], inputs['effect_placed'].to(tokens.dtype)
        batch, area = tokens.shape[0], side*side
        counts = tokens.new_zeros((batch, area)).scatter_add(1, pixels, placed)
        shares = SCATTER_VOLUME/counts.gather(1, pixels).clamp_min(SCATTER_VOLUME)*placed
        values = (F.relu(tokens @ p['effect_scatter_w']+p['effect_scatter_b'])*shares[:, :, None]) @ p['effect_map_w']
        image = values.new_zeros((batch, area, values.shape[-1])).scatter_add(
            1, pixels[:, :, None].expand(-1, -1, values.shape[-1]), values)
        return image.reshape(batch, side, side, -1).permute(0, 3, 1, 2)

    def cell_context(self, inputs, spatial, features, keys):
        """For effects='map': every entity adds the 1x1 map at its cell (what is around it, effects included)
        to its features and keys."""
        if self.config.get('effects') != EFFECTS_MAP:
            return features, keys
        p = self.weights
        cells = entity_cells(self.config, inputs['scatter_pixels'])
        flat = spatial.flatten(2).transpose(1, 2)
        here = flat.gather(1, cells[:, :, None].expand(-1, -1, flat.shape[-1]))
        real = inputs['real'][:, :, None].to(here.dtype)  # Padding and the empty dummy stay zero, as entity_order.
        add = (here @ p['unit_cell_w']+p['unit_cell_b'])*real
        return features+add.to(features.dtype), keys+((here @ p['unit_cell_key_w'])*real).to(keys.dtype)

    def encode_map(self, inputs, scattered, effects=None):
        """TorchPolicyV3.encode_map, also returning the projected skips: ([1x1 map ‖ skips], embedding)."""
        if self.config['network'] != NETWORK:
            return super().encode_map(inputs, scattered, effects)
        p, config = self.weights, self.config
        side = config['map_side']*SCATTER_SCALE//MAP_POOL
        batch, area = scattered.shape[0], side*side
        pixels, real = inputs['scatter_pixels'], inputs['real']
        counts = torch.zeros((batch, area), dtype=scattered.dtype, device=self.device).scatter_add(1, pixels, real.to(scattered.dtype))
        shares = SCATTER_VOLUME/counts.gather(1, pixels).clamp_min(SCATTER_VOLUME)*real
        project = p['map_project'][:, :, 0, 0]
        planes_in = project.shape[1]-scattered.shape[-1]
        values = (scattered*shares[:, :, None]) @ project[:, planes_in:].T
        image = torch.zeros((batch, area, values.shape[-1]), dtype=values.dtype, device=self.device).scatter_add(
            1, pixels[:, :, None].expand(-1, -1, values.shape[-1]), values)
        image = image.reshape(batch, side, side, -1).permute(0, 3, 1, 2)
        static = inputs['map_statics'][inputs['map_static_index']].permute(0, 2, 1)
        visibility = inputs['map_visibility']
        planes = torch.zeros((batch, 5*area), dtype=static.dtype, device=self.device).scatter_add(
            1, inputs['map_channels']*area+pixels, real.to(static.dtype))
        planes = (planes[:, :4*area].reshape(batch, 4, area)+1).log()/4
        attacked = (inputs['numeric'][:, :, UNDER_ATTACK_COLUMN]*real).to(static.dtype)
        alert = torch.zeros((batch, area), dtype=static.dtype, device=self.device).scatter_add(1, pixels, attacked).clamp_max(1)[:, None]
        maps = torch.cat([static, visibility, planes, alert], dim=1).reshape(batch, -1, side, side)
        x = F.relu(F.conv2d(maps.to(image.dtype), p['map_project'][:, :planes_in], p['map_project_b'].flatten())+image)
        x = x.contiguous(memory_format=torch.channels_last)
        for stage in range(MAP_STAGES):
            x = F.relu(self.conv(x, f'map_down{stage}', 2, 1))
        skips = []
        for block in range(MAP_RESIDUAL_BLOCKS):
            skips.append(F.relu(self.conv(F.relu(x), f'map_skip{block}')))
            residual = x
            x = self.conv(F.relu(x), f'map_res{block}_a', padding=1)
            x = self.conv(F.relu(x), f'map_res{block}_b', padding=1)+residual
        out = F.relu(self.conv(F.relu(x), 'map_out'))
        return torch.cat([out, *skips], dim=1), F.relu(out.flatten(1) @ p['map_fc_w']+p['map_fc_b'])

    def location(self, spatial, core, actions, summary, build=None, context=None):
        """TorchPolicyV3.location with a skip added in each FiLM block, deepest map first (AlphaStar).
        For 'movement': movement_location, which needs context (movement_context)."""
        if self.config['network'] == MOVEMENT:
            return self.movement_location(spatial, core, actions, summary, build, context)
        if self.config['network'] != NETWORK:
            return super().location(spatial, core, actions, summary, build)
        p, config = self.weights, self.config
        rows, n, m = spatial.shape[0], config['map_side'], config['map_hidden']
        out = self.widths['map_out']
        skips = spatial[:, out:].reshape(rows, LOCATION_BLOCKS, SKIP, n, n)
        query = self.with_action(core, 'position', actions, summary) @ p['location_query_w']+p['location_query_b']
        x = F.relu(torch.cat([spatial[:, :out].to(query.dtype), query.reshape(rows, 4, n, n)], dim=1))
        x = F.relu(self.conv(x, 'location_input'))
        film = (query @ p['location_film_w']+p['location_film_b'])[:, :, None, None]
        for block in range(LOCATION_BLOCKS):
            residual = F.relu(self.conv(x, f'location_block{block}_a'))
            y = self.conv(residual, f'location_block{block}_b', padding=1)
            gamma, beta = film[:, 2*block*m:(2*block+1)*m]+1, film[:, (2*block+1)*m:(2*block+2)*m]
            skip = self.conv(skips[:, LOCATION_BLOCKS-1-block].to(query.dtype), f'location_skip{block}')
            x = F.relu(F.relu(y*gamma+beta)+residual+skip)
        if build is not None:
            where, features = build
            x = x.index_add(0, where, (features @ p['build_map_w']).permute(0, 2, 1).reshape(len(where), m, n, n))
        y = x
        stages = location_stages(config)
        for stage in range(len(stages)):
            y = F.conv_transpose2d(y, p[f'location_up{stage}'], p[f'location_up{stage}_b'].flatten(), stride=2, padding=1)
            if stage < len(stages)-1:
                y = F.relu(F.layer_norm(y.flatten(1), (y[0].numel(),), eps=1e-5).reshape(y.shape))
        fine = config['position_bins']//n
        flat = y.flatten(1)
        flat = torch.tanh((flat-flat.gather(1, flat.detach().argmax(1, keepdim=True)))/LOGIT_CAP)*LOGIT_CAP
        return flat[:, self.location_order].reshape(rows, n*n, fine*fine)

    def movement_planes(self, query, context):
        """(rows, 1+RELEVANCE_HEADS, n, n): where the selected units are (log 1 + count per cell), and for each
        head the share of attention from the location query to every entity, painted at the entity's cell."""
        p, n = self.weights, self.config['map_side']
        cells, real, keys = context['cells'], context['real'] > 0, context['keys'].to(query.dtype)
        selected, lengths = context['selected'], context['lengths']
        rows, count = cells.shape
        chosen = torch.arange(selected.shape[1], device=cells.device)[None] < lengths[:, None]
        own = query.new_zeros((rows, n*n)).scatter_add(1, cells.gather(1, selected), chosen.to(query.dtype)).log1p()
        q = (query @ p['relevance_q_w']+p['relevance_q_b']).reshape(rows, RELEVANCE_HEADS, RELEVANCE_WIDTH)
        k = (keys @ p['relevance_k_w']).reshape(rows, count, RELEVANCE_HEADS, RELEVANCE_WIDTH)
        scores = torch.einsum('rehd,rhd->rhe', k, q)/math.sqrt(RELEVANCE_WIDTH)
        share = scores.masked_fill(~real[:, None], -torch.inf).softmax(dim=-1).nan_to_num(0.)
        mass = query.new_zeros((rows, RELEVANCE_HEADS, n*n)).scatter_add(
            2, cells[:, None].expand(-1, RELEVANCE_HEADS, -1), share)
        return torch.cat([own[:, None], mass], dim=1).reshape(rows, 1+RELEVANCE_HEADS, n, n)

    def cell_attention(self, x):
        """One pre-norm self-attention block (CELL_HEADS) and MLP over the n x n cells: whole-map reach."""
        p = self.weights
        rows, m, n, _ = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        h = F.layer_norm(tokens+p['cell_position'], (m,), p['cell_attention_norm_g'], p['cell_attention_norm_b'], eps=1e-5)
        heads = CELL_HEADS if m % CELL_HEADS == 0 else 1  # Test models' odd widths take one head.
        q, k, v = (h @ p['cell_qkv_w']+p['cell_qkv_b']).reshape(rows, n*n, 3, heads, m//heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(rows, n*n, m)
        tokens = tokens+attended @ p['cell_out_w']+p['cell_out_b']
        h = F.layer_norm(tokens, (m,), p['cell_mlp_norm_g'], p['cell_mlp_norm_b'], eps=1e-5)
        tokens = tokens+F.relu(h @ p['cell_mlp_a_w']+p['cell_mlp_a_b']) @ p['cell_mlp_b_w']+p['cell_mlp_b_b']
        return tokens.transpose(1, 2).reshape(rows, m, n, n)

    def movement_location(self, spatial, core, actions, summary, build, context):
        """TorchPolicyV3.location, also seeing the selected units and every entity relative to them
        (notes/movement.md): movement_planes beside location_input, cell_attention after the FiLM blocks."""
        p, config = self.weights, self.config
        rows, n, m = spatial.shape[0], config['map_side'], config['map_hidden']
        query = self.with_action(core, 'position', actions, summary) @ p['location_query_w']+p['location_query_b']
        x = F.relu(torch.cat([spatial.to(query.dtype), query.reshape(rows, 4, n, n)], dim=1))
        x = F.relu(self.conv(x, 'location_input')+self.conv(self.movement_planes(query, context), 'location_extra'))
        film = (query @ p['location_film_w']+p['location_film_b'])[:, :, None, None]
        for block in range(LOCATION_BLOCKS):
            residual = F.relu(self.conv(x, f'location_block{block}_a'))
            y = self.conv(residual, f'location_block{block}_b', padding=1)
            gamma, beta = film[:, 2*block*m:(2*block+1)*m]+1, film[:, (2*block+1)*m:(2*block+2)*m]
            x = F.relu(F.relu(y*gamma+beta)+residual)
        x = F.relu(self.cell_attention(x))
        if build is not None:
            where, features = build
            x = x.index_add(0, where, (features @ p['build_map_w']).permute(0, 2, 1).reshape(len(where), m, n, n))
        y = x
        stages = location_stages(config)
        for stage in range(len(stages)):
            y = F.conv_transpose2d(y, p[f'location_up{stage}'], p[f'location_up{stage}_b'].flatten(), stride=2, padding=1)
            if stage < len(stages)-1:
                y = F.relu(F.layer_norm(y.flatten(1), (y[0].numel(),), eps=1e-5).reshape(y.shape))
        fine = config['position_bins']//n
        flat = y.flatten(1)
        flat = torch.tanh((flat-flat.gather(1, flat.detach().argmax(1, keepdim=True)))/LOGIT_CAP)*LOGIT_CAP
        return flat[:, self.location_order].reshape(rows, n*n, fine*fine)


def entity_cells(config, pixels):
    """Each entity's location cell (row-major n x n, as the location head's maps) from its scatter pixel."""
    side = config['map_side']*SCATTER_SCALE//MAP_POOL
    step = side//config['map_side']
    return (pixels//side)//step*config['map_side']+(pixels % side)//step


def movement_context(model, bundle, rows, keys, selected, lengths):
    """movement_location's context for these batch rows: entity cells, real mask, keys and the selection
    (indices into the row's entities, and their counts)."""
    return {'cells': entity_cells(model.config, bundle['pixels'][rows]), 'real': bundle['real'][rows], 'keys': keys,
            'selected': selected, 'lengths': lengths}
