"""Batched PyTorch AlphaGenerals policy over the NumPy reference parameters and inputs.

Weights keep the reference parameter names and layout, so checkpoints are shared with
generals_policy.NeuralPolicy, whose one-observation equations these batch
(test_alphagenerals holds them equal). Padding never enters attention or pooling.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from generals_policy import (ARCHITECTURE, ATTENTION_LAYERS, GROUPS, LOCATION_BLOCKS, LOGIT_CAP, LSTM_LAYERS,
                             MAP_POOL, MAP_RESIDUAL_BLOCKS, MAP_STAGES, SCATTER_SCALE, SCATTER_VOLUME, SELECTED_UNIT_DIVISOR,
                             UNDER_ATTACK_COLUMN, UNIT_GROUPS, group_sources, location_order, location_stages, token_groups,
                             widths)


class TorchPolicyV3(nn.Module):
    def __init__(self, reference, *, device='cpu', dtype=torch.float64):
        super().__init__()
        if reference.config.get('architecture') != ARCHITECTURE:
            raise ValueError('PyTorch learner requires the AlphaGenerals policy')
        self.reference, self.config = reference, reference.config
        self.widths = widths(self.config)
        # The transformer's token groups: the unit groups, plus effect tokens unless effects are painted on the map.
        self.groups = token_groups(self.config)
        self.weights = nn.ParameterDict(OrderedDict((name, nn.Parameter(torch.tensor(
            value.data, dtype=dtype, device=device))) for name, value in reference.parameters.items()))
        identity = reference.encoder.identity
        for name in ('families', 'base_factions', 'generals', 'type_objects', 'template_objects'):
            self.register_buffer('identity_' + name, torch.tensor(
                getattr(identity, name), dtype=torch.long, device=device))
        self.register_buffer('location_order', torch.tensor(location_order(self.config).reshape(-1), dtype=torch.long, device=device))
        # The kind of each flat action (reports compare kinds).
        self.register_buffer('action_kinds', torch.tensor(reference.actions[:, 0], dtype=torch.long, device=device))
        from torch_training_inputs import TensorStager
        self.stager = TensorStager(self)
        self.legality_key = None

    @property
    def device(self):
        return next(iter(self.weights.values())).device

    @property
    def dtype(self):
        return next(iter(self.weights.values())).dtype

    def array(self, values, *, dtype=None):
        # Cache arrays are immutable; copying also avoids PyTorch alias warnings.
        return torch.tensor(np.asarray(values), device=self.device, dtype=dtype or self.dtype)

    def indices(self, values):
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def object_embedding(self, objects):
        p = self.weights
        # Embedding backward has a specialised reduction for repeated IDs. The
        # generic advanced-index scatter is costly for mostly-empty queues and
        # the many entities sharing a faction/general identity.
        return (F.embedding(self.identity_families[objects], p['family_embedding'])
                + F.embedding(self.identity_base_factions[objects], p['base_faction_embedding'])
                + F.embedding(self.identity_generals[objects], p['general_embedding'])
                + F.embedding(objects, p['variant_embedding']))

    def type_embedding(self, types):
        return self.object_embedding(self.identity_type_objects[types])

    def regularization_loss(self):
        if not self.config.get('variant_l2', 0):
            return torch.zeros((), dtype=self.dtype, device=self.device)
        return sum(self.weights[name].square().sum() for name in
                   ('general_embedding', 'variant_embedding')) * self.config['variant_l2']

    @staticmethod
    def normalize(value):
        return F.layer_norm(value, (value.shape[-1],), eps=1e-5)

    def layer_norm(self, value, name):
        return F.layer_norm(value, (value.shape[-1],), self.weights[name+'_g'], self.weights[name+'_b'], eps=1e-5)

    @property
    def float_dtype(self):
        return np.float32 if self.dtype == torch.float32 else np.float64

    @property
    def device_type(self):
        return self.device.type

    def conv(self, values, name, stride=1, padding=0):
        return F.conv2d(values, self.weights[name], self.weights[name+'_b'].flatten(), stride=stride, padding=padding)

    # Encoder (plan §1a-1d), one row per observation.

    def entity_inputs(self, inputs):
        """(units (batch, count, d), effect tokens (batch, effects, d)): NeuralPolicy._entity_inputs."""
        p, config = self.weights, self.config
        features = inputs['numeric']
        batch, count, _ = features.shape
        # One per-type table for an entity's own type and its queue items, then a single input
        # GEMM: the same sums as separate per-slice products.
        table = self.object_embedding(self.identity_type_objects)
        types = F.embedding(inputs['types'], table)
        # Queue: slot 0 is in production now, the rest pending; a slot is a unit type (-1: none)
        # or an upgrade (0: none), as NeuralPolicy.queue_inputs.
        queue, queued_upgrades = inputs['queue_units'], inputs['queue_upgrades']
        items = (F.embedding(queue.clamp(min=0), table)*(queue >= 0)[..., None].to(self.dtype)
                 + F.embedding(queued_upgrades, p['upgrade_embedding'])*(queued_upgrades > 0)[..., None].to(self.dtype))
        types = torch.cat([types, items[:, :, 0], items[:, :, 1:].sum(dim=2)], dim=-1)
        # Empty upgrade slots (index 0) are masked out, as in the reference.
        slots = inputs['entity_upgrades']
        upgrades = (F.embedding(slots, p['upgrade_embedding']) * (slots > 0)[..., None].to(self.dtype)).sum(dim=2)
        # Per-button state, as generals_policy.entity_command_inputs: ready, or charging plus progress.
        ids, states = inputs['entity_commands'], inputs['entity_command_states']
        ready, progress = states[..., :1], states[..., 1:]
        charging = (1-ready)*(ids > 0)[..., None].to(self.dtype)
        commands = (F.embedding(ids, p['entity_command_ready_embedding'])*ready
                    + (F.embedding(ids, p['entity_command_charging_embedding'])
                       + F.embedding(ids, p['entity_command_progress_embedding'])*progress)*charging).sum(dim=2)
        entity = torch.tanh(torch.cat([features, types, upgrades, commands], dim=-1) @ p['entity_w'] + p['entity_b'])
        from unit_tasks import RELATIONS
        hidden = self.widths['entity']
        base = entity.reshape(batch*count, hidden)
        rows = inputs['task_rows'].numel()
        # Masking a linked row before its product equals masking after (0/1);
        # all relations then share one stacked GEMM.
        linked = (base.index_select(0, inputs['task_relations'].reshape(-1)).reshape(rows, len(RELATIONS), hidden)
                  * inputs['task_relation_valid'][:, :, None])
        addition = (inputs['task_numeric'] @ p['task_w']
                    + linked.flatten(1) @ torch.cat([p['task_'+role+'_w'] for role in RELATIONS])
                    + inputs['task_relation_features'].flatten(1)
                    @ torch.cat([p['task_'+role+'_features_w'] for role in RELATIONS]))
        entity = torch.tanh(base.index_add(0, inputs['task_rows'], addition.to(base.dtype))).reshape(batch, count, hidden)
        tokens = torch.tanh(inputs['effect_numeric'] @ p['effect_w'] + self.type_embedding(inputs['effect_types']) @ p['effect_type_w']
                            + F.embedding(inputs['effect_powers'], p['power_embedding']) @ p['effect_power_w'] + p['effect_b'])
        return entity, tokens.to(entity.dtype)

    def attend(self, q, k, v, some, mask):
        """Masked multi-head attention of (batch, heads, queries, width) over (batch, heads, sources, width).

        mask: (batch, 1, 1, sources), with every entry set for a row whose source group is empty;
        some: (batch, 1, 1) whether it has any, so such a row gets zeros, as the reference."""
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)*some[..., None].to(q.dtype)

    def group_transformer(self, groups, valid):
        """Per-group features after the plan's layers; groups[g] is (batch, n_g, d), valid[g] (batch, n_g).

        One projection per group carries its queries to every source and its keys and values for every
        query group; each (query group, source group) pair is one fused attention."""
        p, x, names = self.weights, list(groups), self.groups
        heads, width = self.widths['heads'], self.widths['head_width']
        G = range(len(names))
        some = [value.any(dim=-1)[:, None, None] for value in valid]
        masks = [(value | ~present[:, 0])[:, None, None] for value, present in zip(valid, some)]
        for layer in range(ATTENTION_LAYERS):
            pair = lambda g, s, kind: p[f'attn{layer}_{names[g]}_{names[s]}_{kind}']
            projected = []
            for g, value in enumerate(x):
                batch, n, _ = value.shape
                weight = torch.cat([pair(g, s, 'q') for s in G]+[pair(q, g, kind) for kind in ('k', 'v') for q in G], dim=1)
                # (batch, n, [q for each source | k for each query group | v for each query group], heads, width)
                out = self.layer_norm(value, f'attn{layer}_{names[g]}_ln') @ weight
                projected.append(out.reshape(batch, n, 3*len(names), heads, width).permute(2, 0, 3, 1, 4))
            updated = []
            for g, value in enumerate(x):
                name, ffn = f'attn{layer}_{names[g]}', f'ffn{layer}_{names[g]}'
                parts = [self.attend(projected[g][s], projected[s][len(names)+g], projected[s][2*len(names)+g], some[s], masks[s])
                         for s in group_sources(g, len(names))]
                parts = torch.cat(parts, dim=1).transpose(1, 2).flatten(2)  # (batch, n, blocks*heads*width)
                value = value+parts @ p[name+'_out_w']+p[name+'_out_b']
                hidden = F.relu(self.layer_norm(value, ffn+'_ln') @ p[ffn+'_in_w']+p[ffn+'_in_b'])
                updated.append(value+hidden @ p[ffn+'_out_w']+p[ffn+'_out_b'])
            x = updated
        return [self.layer_norm(value, f'final_{names[g]}_ln') for g, value in enumerate(x)]

    def pool(self, value, valid, group):
        """Attention pooling (NeuralPolicy._pool): (batch, queries*heads*width); an empty group gives zeros."""
        p, W = self.weights, self.widths
        heads, width, queries = W['heads'], W['head_width'], W['pool_queries']
        batch, count = value.shape[:2]
        k = (value @ p[f'pool_{group}_k']).reshape(batch, count, heads, width)
        v = (value @ p[f'pool_{group}_v']).reshape(batch, count, heads, width)
        q = p[f'pool_{group}_query'].reshape(queries, heads, width)
        scores = torch.einsum('bnhw,qhw->bqhn', k, q.to(k.dtype))/math.sqrt(width)
        mask = valid[:, None, None, :]
        scores = scores.masked_fill(~mask, -torch.inf)
        scores = torch.where(mask.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(dim=-1)*mask
        return torch.einsum('bqhn,bnhw->bqhw', weights, v).reshape(batch, queries*heads*width)

    @staticmethod
    def entity_order(values, index, valid, count):
        """Per-group rows (batch, n_g, width) back in entity order (batch, count, width); padding stays zero."""
        width = values[0].shape[-1]
        result = values[0].new_zeros((values[0].shape[0], count, width))
        for value, rows, mask in zip(values, index, valid):
            result = result.scatter_add(1, rows[:, :, None].expand(-1, -1, width), value*mask[:, :, None].to(value.dtype))
        return result

    def effect_image(self, inputs, tokens):
        """Effects painted beside the scattered units, (batch, scatter, side, side), or None: effect tokens
        go through the transformer instead (network_alphastar paints them for config effects='map')."""
        return None

    def cell_context(self, inputs, spatial, features, keys):
        """(features, keys) of every entity after the map: unchanged here (network_alphastar adds what the
        map holds at each unit's cell for config effects='map')."""
        return features, keys

    def encode_map(self, inputs, scattered, effects=None):
        """(1x1 map (batch, map_out, n, n), embedding): NeuralPolicy._encode_map; effects: effect_image."""
        p, config = self.weights, self.config
        side = config['map_side']*SCATTER_SCALE//MAP_POOL
        batch, area = scattered.shape[0], side*side
        pixels, real = inputs['scatter_pixels'], inputs['real']
        counts = torch.zeros((batch, area), dtype=scattered.dtype, device=self.device).scatter_add(1, pixels, real.to(scattered.dtype))
        shares = SCATTER_VOLUME/counts.gather(1, pixels).clamp_min(SCATTER_VOLUME)*real
        # map_project is a 1x1 convolution of (planes, image of scattered units), so its image part applies to
        # each unit's values before they are scattered: the same sums without building the concatenated image.
        project = p['map_project'][:, :, 0, 0]
        planes_in = project.shape[1]-scattered.shape[-1]
        values = (scattered*shares[:, :, None]) @ project[:, planes_in:].T
        image = torch.zeros((batch, area, values.shape[-1]), dtype=values.dtype, device=self.device).scatter_add(
            1, pixels[:, :, None].expand(-1, -1, values.shape[-1]), values)
        image = image.reshape(batch, side, side, -1).permute(0, 3, 1, 2)
        if effects is not None:
            image = image+effects.to(image.dtype)
        # The same planes as generals_policy.map_planes, built here from compact inputs.
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
        for block in range(MAP_RESIDUAL_BLOCKS):
            residual = x
            x = self.conv(F.relu(x), f'map_res{block}_a', padding=1)
            x = self.conv(F.relu(x), f'map_res{block}_b', padding=1)+residual
        out = F.relu(self.conv(F.relu(x), 'map_out'))
        return out, F.relu(out.flatten(1) @ p['map_fc_w']+p['map_fc_b'])

    def scalars(self, inputs):
        """NeuralPolicy._scalars over a batch, in scalar_input_size's order."""
        p = self.weights
        identities = inputs['identities']
        identity = p['base_faction_embedding'][identities[:, 0]]+p['general_embedding'][identities[:, 1]]
        identities = inputs['enemy_identities']
        enemy = p['base_faction_embedding'][identities[:, 0]]+p['general_embedding'][identities[:, 1]]
        batch = identity.shape[0]
        counts = torch.zeros((batch, len(self.config['types'])), dtype=self.dtype, device=self.device).scatter_add(
            1, inputs['types'], inputs['real'])
        parts = [inputs['scalars'], identity, enemy, inputs['ratings'], inputs['apm'], inputs['opener'], inputs['player_flags'],
                 counts.sqrt(), inputs['available_actions']]
        for i in range(2):
            template = self.type_embedding(inputs['task_command_templates'][:, i])
            parts += [inputs['task_commands'][:, i], template*inputs['task_command_template_present'][:, i, None]]
        x = F.relu(torch.cat([part.to(identity.dtype) for part in parts], dim=1) @ p['scalar_in_w']+p['scalar_in_b'])
        return F.relu(x @ p['scalar_hidden_w']+p['scalar_hidden_b'])

    def encode_inputs(self, inputs, selection=None):
        """Encoder outputs per row: core input, unit features, keys, the 1x1 map and, given
        selection = (rows, indices, lengths), the selected units' [Mean, Max] summary."""
        p = self.weights
        entity, tokens = self.entity_inputs(inputs)
        count, names = entity.shape[1], self.groups
        flat = torch.cat([entity, tokens], dim=1) if len(names) > UNIT_GROUPS else entity
        index = [inputs[f'group{g}_index'] for g in range(len(names))]
        valid = [inputs[f'group{g}_valid'] for g in range(len(names))]
        groups = [flat.gather(1, rows[:, :, None].expand(-1, -1, flat.shape[-1])) for rows in index]
        final = self.group_transformer(groups, valid)
        units = final[:UNIT_GROUPS]
        features = self.entity_order(units, index[:UNIT_GROUPS], valid[:UNIT_GROUPS], count)
        keys = self.entity_order([value @ p[f'key_{names[g]}_w']+p[f'key_{names[g]}_b'] for g, value in enumerate(units)],
                                 index[:UNIT_GROUPS], valid[:UNIT_GROUPS], count)
        scattered = self.entity_order([F.relu(value @ p[f'scatter_{names[g]}_w']+p[f'scatter_{names[g]}_b'])
                                       for g, value in enumerate(units)], index[:UNIT_GROUPS], valid[:UNIT_GROUPS], count)
        spatial, map_embedding = self.encode_map(inputs, scattered, self.effect_image(inputs, tokens))
        features, keys = self.cell_context(inputs, spatial, features, keys)
        pools = [self.pool(value, valid[g], names[g]) for g, value in enumerate(final)]
        if len(names) < len(GROUPS):  # Effects on the map: their pool reads the tokens without the transformer.
            pools.append(self.pool(tokens, inputs['effect_valid'], GROUPS[-1]))
        core_input = torch.cat([self.scalars(inputs).to(map_embedding.dtype), *[pool.to(map_embedding.dtype) for pool in pools],
                                map_embedding], dim=1)
        result = {'core_input': core_input, 'features': features, 'keys': keys, 'map': spatial}
        if selection is not None:
            result['summary'] = self.selection_summary(features, *selection)
        return result

    @staticmethod
    def selection_summary(features, rows, indices, lengths):
        """[Mean, Max] of each row's selected units' features; zeros for rows without a selection.

        rows (k,) index the batch; indices (k, longest) and lengths (k,) are their selections."""
        batch, _, width = features.shape
        summary = features.new_zeros((batch, 2*width))
        if not rows.numel():
            return summary
        chosen = features[rows[:, None], indices]
        mask = torch.arange(indices.shape[1], device=features.device)[None, :] < lengths[:, None]
        total = (chosen*mask[:, :, None].to(chosen.dtype)).sum(dim=1)
        mean = total/lengths.clamp_min(1)[:, None].to(chosen.dtype)
        peak = chosen.masked_fill(~mask[:, :, None], -torch.inf).amax(dim=1)
        has = (lengths > 0)[:, None]
        value = torch.cat([torch.where(has, mean, torch.zeros_like(mean)), torch.where(has, peak, torch.zeros_like(peak))], dim=1)
        return summary.index_copy(0, rows, value)

    # Core (plan §1e): FC-ReLU, then the residual LSTM over each lane's decisions.

    def initial_state(self, lanes):
        size = self.widths['core']
        return [(torch.zeros((lanes, size), dtype=self.dtype, device=self.device),
                 torch.zeros((lanes, size), dtype=self.dtype, device=self.device)) for _ in range(LSTM_LAYERS)]

    def lstm_cell(self, layer, x, h, c, projected=None):
        """One layer-normed LSTM step (NeuralPolicy._core); x @ W_x may be given precomputed."""
        p, size = self.weights, self.widths['core']
        pre = (x @ p[f'core_lstm{layer}_x_w'] if projected is None else projected)+h @ p[f'core_lstm{layer}_h_w']
        gates = F.layer_norm(pre.reshape(-1, 4, size), (size,), eps=1e-5)*p[f'core_lstm{layer}_gate_g']+p[f'core_lstm{layer}_gate_b']
        c = torch.sigmoid(gates[:, 1])*c+torch.sigmoid(gates[:, 0])*torch.tanh(gates[:, 2])
        h = torch.sigmoid(gates[:, 3])*torch.tanh(self.layer_norm(c, f'core_lstm{layer}_cell'))
        return h, c

    def core(self, core_input, resets, state):
        """(core outputs (lanes, steps, W), state after the last step) in fp32.

        core_input (lanes, steps, inputs); resets (lanes, steps) zero the state before that step
        (a new episode). state: the LSTM (h, c) per layer, carried from the lane's previous unroll."""
        p = self.weights
        with torch.autocast(self.device.type, enabled=False):
            u = F.relu(core_input.to(self.dtype) @ p['core_in_w']+p['core_in_b'])
            projected = u @ p['core_lstm0_x_w']  # The first layer's input is known for every step: one GEMM.
            keep = (~resets).to(u.dtype)
            outputs, state = [], list(state)
            for step in range(u.shape[1]):
                x = None
                for layer in range(LSTM_LAYERS):
                    h, c = (value*keep[:, step, None] for value in state[layer])
                    h, c = self.lstm_cell(layer, x, h, c, projected[:, step] if layer == 0 else None)
                    state[layer] = (h, c)
                    x = h
                outputs.append(u[:, step]+x)
            return torch.stack(outputs, dim=1), state

    def values(self, core):
        p = self.weights
        return (core @ p['value_w']+p['value_b']).squeeze(-1)

    # Heads (plan §1f): every head reads the core output and its own embedding of the selected action.

    def with_action(self, core, head, actions, summary=None):
        parts = [core, F.embedding(actions, self.weights[head+'_action_embedding']).to(core.dtype)]
        return torch.cat(parts+([summary.to(core.dtype)] if summary is not None else []), dim=-1)

    def action_logits(self, core):
        return core @ self.weights['action_w']+self.weights['action_b']

    def slot_logits(self, core, actions, summary):
        return self.with_action(core, 'slot', actions, summary) @ self.weights['slot_w']+self.weights['slot_b']

    def angle_logits(self, core, actions):
        return self.with_action(core, 'angle', actions) @ self.weights['angle_w']+self.weights['angle_b']

    def target_logits(self, core, actions, summary, keys):
        """(rows, count): conditioned concat-attention over every entity's key."""
        p = self.weights
        query = self.with_action(core, 'target', actions, summary) @ p['target_in_w']+p['target_in_b']
        hidden = torch.tanh(keys.to(query.dtype) @ p['target_score_k_w']+query[:, None])
        return (hidden @ F.embedding(actions, p['target_action_embedding'])[:, :, None]).squeeze(-1)

    def unit_context(self, core, actions):
        p = self.weights
        return self.with_action(core, 'unit', actions) @ p['unit_in_w']+p['unit_in_b']

    def unit_query(self, context):
        p = self.weights
        return F.relu(context) @ p['unit_query_w']+p['unit_query_b']

    def unit_step(self, query, state):
        """One step of the unit query LSTM (torch.lstm's equations; NeuralPolicy._lstm_step)."""
        p = self.weights
        h, c = state
        gates = query @ p['unit_lstm_input_w']+h @ p['unit_lstm_hidden_w']+p['unit_lstm_b']
        i, f, g, o = gates.chunk(4, dim=-1)
        c = torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c), c

    def unit_keys(self, keys):
        """Every entity key then the end token's, projected for the unit pointer: (rows, count+1, a)."""
        p = self.weights
        end = p['unit_end_key'].to(keys.dtype).expand(keys.shape[0], 1, -1)
        return torch.cat([keys, end], dim=1) @ p['unit_score_k_w']

    def unit_logits(self, hidden, projected_keys, actions):
        """Scores of each token's query over its row's projected keys and the end token."""
        p = self.weights
        scores = torch.tanh(projected_keys+(hidden @ p['unit_score_q_w']+p['unit_score_b'])[:, None])
        return (scores @ F.embedding(actions, p['unit_action_embedding'])[:, :, None]).squeeze(-1)

    def unit_add(self, context, key):
        p = self.weights
        return context+(key/SELECTED_UNIT_DIVISOR) @ p['unit_ar_w']+p['unit_ar_b']

    def location(self, spatial, core, actions, summary, build=None):
        """(rows, n*n, offsets) position logits: NeuralPolicy._location over a batch."""
        p, config = self.weights, self.config
        rows, n, m = spatial.shape[0], config['map_side'], config['map_hidden']
        query = self.with_action(core, 'position', actions, summary) @ p['location_query_w']+p['location_query_b']
        x = F.relu(torch.cat([spatial.to(query.dtype), query.reshape(rows, 4, n, n)], dim=1))
        x = F.relu(self.conv(x, 'location_input'))
        film = (query @ p['location_film_w']+p['location_film_b'])[:, :, None, None]
        for block in range(LOCATION_BLOCKS):
            residual = F.relu(self.conv(x, f'location_block{block}_a'))
            y = self.conv(residual, f'location_block{block}_b', padding=1)
            gamma, beta = film[:, 2*block*m:(2*block+1)*m]+1, film[:, (2*block+1)*m:(2*block+2)*m]
            x = F.relu(F.relu(y*gamma+beta)+residual)
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

    def encode_batch(self, encoded):
        """Stage encoded rows (live play) and run the encoder."""
        from training_preparation import encode_values
        values, _ = encode_values(self, encoded)
        staged = self.stager.stage(values)
        out = self.encode_inputs(staged)
        # Entity positions for the movement network's location head (network_alphastar.movement_context).
        out['pixels'], out['real'] = staged['scatter_pixels'], staged['real']
        return out

    def events(self, encoded=None, *, prepared=None):
        """(core outputs (rows, W), encoder outputs) of independent observations, each scored as its
        game's first decision (tests and tools; training and play carry the LSTM state). prepared:
        a training_preparation microbatch instead of encoded rows."""
        out = (self.encode_batch(encoded) if prepared is None
               else self.encode_inputs(self.stager.stage(prepared['values'])))
        rows = out['core_input'].shape[0]
        core, _ = self.core(out['core_input'][:, None], torch.zeros((rows, 1), dtype=torch.bool, device=self.device),
                            self.initial_state(rows))
        return core[:, 0], out

    def teacher_heads(self, encoded, teachers, core, bundle, *, loss_only=False, plan=None, position_metrics=False):
        """torch_training_inputs.teacher_heads for events()' outputs, planning the teachers here."""
        from torch_training_inputs import teacher_heads
        from training_preparation import teacher_plan
        plan = plan or teacher_plan(self, encoded, teachers, bundle['keys'].shape[1])
        plan['staged'] = staged = self.stager.stage(plan['values'])
        summary = self.selection_summary(bundle['features'], staged['command_rows'], staged['selected_indices'],
                                         staged['selected_lengths'])
        return teacher_heads(self, core, {**bundle, 'summary': summary}, plan, loss_only=loss_only,
                             position_metrics=position_metrics)


def torch_policy(reference, **options):
    """The PyTorch module for a reference policy: AlphaGenerals, or a temporary comparison network."""
    if reference.config.get('network'):
        from network_alphastar import AlphaStarTorchPolicy
        return AlphaStarTorchPolicy(reference, **options)
    return TorchPolicyV3(reference, **options)
