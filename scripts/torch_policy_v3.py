"""Batched PyTorch event policy over the NumPy reference parameters and inputs.

Weights retain the reference parameter names and layout, so checkpoints are shared
with the deployed NumPy policy. Padding never enters attention or pooling.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from generals_policy import (ATTENTION_LAYERS, ARCHITECTURE, LOCATION_BLOCKS, MAP_RESIDUAL_BLOCKS, SCATTER_SCALE,
                             SCATTER_VOLUME, UNDER_ATTACK_COLUMN, location_order, location_stages)


class TorchPolicyV3(nn.Module):
    def __init__(self, reference, *, device='cpu', dtype=torch.float64):
        super().__init__()
        if reference.config.get('architecture') != ARCHITECTURE:
            raise ValueError('PyTorch learner requires the event policy')
        self.reference, self.config = reference, reference.config
        self.weights = nn.ParameterDict(OrderedDict((name, nn.Parameter(torch.tensor(
            value.data, dtype=dtype, device=device))) for name, value in reference.parameters.items()))
        identity = reference.encoder.identity
        for name in ('families', 'base_factions', 'generals', 'type_objects', 'template_objects'):
            self.register_buffer('identity_' + name, torch.tensor(
                getattr(identity, name), dtype=torch.long, device=device))
        self.register_buffer('location_order', torch.tensor(location_order(self.config).reshape(-1), dtype=torch.long, device=device))
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
                   ('general_embedding', 'variant_embedding', 'template_b')) * self.config['variant_l2']

    @staticmethod
    def normalize(value):
        return F.layer_norm(value, (value.shape[-1],), eps=1e-5)

    @property
    def float_dtype(self):
        return np.float32 if self.dtype == torch.float32 else np.float64

    @property
    def device_type(self):
        return self.device.type

    def encode_batch(self, encoded):
        """Pad only this chunk's entity dimension; padding never enters attention/pooling."""
        from training_preparation import encode_values
        values, _ = encode_values(self, encoded)
        return self.encode_inputs(self.stager.stage(values))

    def conv(self, values, name, stride=1, padding=0):
        return F.conv2d(values, self.weights[name], stride=stride, padding=padding)+self.weights[name+'_b']

    def encode_map(self, inputs, entity):
        """(skips: (batch, 5, m, n, n), embedding): AlphaStar's scatter and spatial encoder."""
        p, config = self.weights, self.config
        n, side = config['map_side'], config['map_side']*SCATTER_SCALE
        batch, area = entity.shape[0], side*side
        pixels, real = inputs['scatter_pixels'], inputs['real']
        counts = torch.zeros((batch, area), dtype=entity.dtype, device=self.device).scatter_add(1, pixels, real.to(entity.dtype))
        shares = SCATTER_VOLUME/counts.gather(1, pixels).clamp_min(SCATTER_VOLUME)*real
        values = F.relu(entity @ p['scatter_w'])*shares[:, :, None]
        image = torch.zeros((batch, area, values.shape[-1]), dtype=values.dtype, device=self.device).scatter_add(
            1, pixels[:, :, None].expand(-1, -1, values.shape[-1]), values)
        image = image.reshape(batch, side, side, -1).permute(0, 3, 1, 2)
        # The same planes as generals_policy.map_planes, built here from compact inputs.
        static = inputs['map_statics'][inputs['map_static_index']].permute(0, 2, 1)
        visibility = F.one_hot(inputs['map_visibility'], 3).permute(0, 2, 1).to(static.dtype)
        planes = torch.zeros((batch, 5*area), dtype=static.dtype, device=self.device).scatter_add(
            1, inputs['map_channels']*area+pixels, real.to(static.dtype))
        planes = (planes[:, :4*area].reshape(batch, 4, area)+1).log()/4
        attacked = (inputs['numeric'][:, :, UNDER_ATTACK_COLUMN]*real).to(static.dtype)
        alert = torch.zeros((batch, area), dtype=static.dtype, device=self.device).scatter_add(1, pixels, attacked).clamp_max(1)[:, None]
        maps = torch.cat([static, visibility, planes, alert], dim=1).reshape(batch, -1, side, side)
        x = F.relu(self.conv(torch.cat([maps.to(image.dtype), image], dim=1), 'map_project'))
        for stage in range(3):
            x = F.relu(self.conv(x, f'map_down{stage}', 2, 1))
        skips = [x]
        for block in range(MAP_RESIDUAL_BLOCKS):
            residual = x
            x = self.conv(F.relu(x), f'map_res{block}_a', padding=1)
            x = self.conv(F.relu(x), f'map_res{block}_b', padding=1)+residual
            skips.append(x)
        embedded = F.relu(F.relu(x).flatten(1) @ p['map_fc']+p['map_fc_b'])
        return torch.stack(skips, dim=1), embedded

    def location(self, spatial, context, build=None):
        """(cell features (rows, m, n, n), logits (rows, n*n, offsets)) for position arguments."""
        p, config = self.weights, self.config
        rows, n, m = spatial.shape[0], config['map_side'], config['map_hidden']
        x = F.relu(torch.cat([spatial[:, MAP_RESIDUAL_BLOCKS],
                              (context @ p['location_context_w']).reshape(rows, 4, n, n)], dim=1))
        x = F.relu(self.conv(x, 'location_input'))
        film = (context @ p['location_film_w']+p['location_film_b'])[:, :, None, None]
        for block in range(LOCATION_BLOCKS):
            residual = F.relu(self.conv(x, f'location_block{block}_a'))
            y = self.conv(residual, f'location_block{block}_b', padding=1)
            gamma, beta = film[:, 2*block*m:(2*block+1)*m]+1, film[:, (2*block+1)*m:(2*block+2)*m]
            x = F.relu(F.relu(y*gamma+beta)+residual+spatial[:, MAP_RESIDUAL_BLOCKS-1-block])
        if build is not None:
            where, features = build
            x = x.index_add(0, where, (features @ p['build_map_w']).permute(0, 2, 1).reshape(len(where), m, n, n))
        cells, y = x, x
        stages = location_stages(config)
        for stage in range(len(stages)):
            y = F.conv_transpose2d(y, p[f'location_up{stage}'], stride=2, padding=1)+p[f'location_up{stage}_b']
            if stage < len(stages)-1:
                y = F.relu(y)
        fine = config['position_bins']//n
        return cells, y.flatten(1)[:, self.location_order].reshape(rows, n*n, fine*fine)

    def encode_inputs(self, inputs):
        """Numerical encoder after CPU preparation and device staging."""
        p, config = self.weights, self.config
        features = inputs['numeric']
        batch, count, _ = features.shape
        # One lookup into a per-type table for current and queued types, then a
        # single input GEMM: the same sums as separate per-slice products, with
        # far fewer token-sized embedding backward passes.
        table = self.object_embedding(self.identity_type_objects)
        types = F.embedding(torch.stack([inputs['types'], inputs['queued']], dim=-1), table).reshape(batch, count, -1)
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
        hidden = config['entity_hidden']
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
        heads, width = config['attention_heads'], config['entity_hidden']//config['attention_heads']
        valid = ~inputs['valid'][:, None, None, :]
        for layer in range(ATTENTION_LAYERS):
            q, k, v = [(entity @ p[f'{name}{layer}']).reshape(batch, count, heads, width).transpose(1, 2)
                       for name in ('attn_q', 'attn_k', 'attn_v')]
            scores = (q @ k.transpose(-1, -2)/math.sqrt(width)).masked_fill(valid, -torch.inf)
            attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, count, config['entity_hidden'])
            entity = self.normalize(entity + attended @ p[f'attn_out{layer}'])
            entity = self.normalize(entity + F.relu(entity @ p[f'entity_ff_in{layer}']) @ p[f'entity_ff_out{layer}'])
        spatial, map_embedding = self.encode_map(inputs, entity)
        scalar = inputs['scalars'] @ p['scalar_w'] + p['scalar_b']
        for i, prefix in enumerate(('task_command', 'task_validation')):
            scalar = scalar + inputs['task_commands'][:, i] @ p[prefix+'_w']
            embedding = self.type_embedding(inputs['task_command_templates'][:, i])
            scalar = scalar + (embedding @ p[prefix+'_template_w']) * inputs['task_command_template_present'][:, i, None]
        identities = inputs['identities']
        identity = p['base_faction_embedding'][identities[:, 0]]+p['general_embedding'][identities[:, 1]]
        scalar = scalar + identity @ p['player_identity_w']
        identities = inputs['enemy_identities']
        enemy = p['base_faction_embedding'][identities[:, 0]]+p['general_embedding'][identities[:, 1]]
        scalar = scalar + enemy @ p['enemy_identity_w']
        scalar = scalar + inputs['ratings'] @ p['ratings_w']
        scalar = scalar + inputs['player_flags'] @ p['player_flags_w']
        counts = torch.zeros((batch, len(config['types'])), dtype=self.dtype, device=self.device).scatter_add(
            1, inputs['types'], inputs['real'])
        available = inputs['available_actions']
        scalar = scalar + counts.sqrt() @ p['unit_counts_w'] + available @ p['available_actions_w']
        gate = F.relu(identity @ p['gate_identity_w'] + enemy @ p['gate_enemy_w'] + available @ p['gate_actions_w'] + p['gate_b'])
        scalar = torch.tanh(scalar)
        query = scalar @ p['pool_query']
        def pool(mask):
            logits = torch.bmm(entity, query.unsqueeze(-1)).squeeze(-1).masked_fill(~mask, -torch.inf)
            logits = torch.where(mask.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
            weights = logits.softmax(dim=1)*mask
            return torch.bmm(weights.unsqueeze(1), entity).squeeze(1)
        features = torch.cat([scalar, pool(inputs['pool_own']), pool(inputs['pool_visible']),
                              pool(inputs['pool_hidden']), pool(inputs['pool_neutral']), map_embedding], dim=1)
        return features, entity, spatial, gate

    def events(self, encoded=None, *, prepared=None):
        """Trunk outputs from encoded rows, or from training_preparation's prepared arrays.

        The first holds the core's output followed by scalar_context.
        """
        if prepared is None:
            features, entities, spatial, gate = self.encode_batch(encoded)
        else:
            features, entities, spatial, gate = self.encode_inputs(self.stager.stage(prepared['values']))
        p = self.weights
        hidden = torch.tanh(torch.tanh(features @ p['event_input_w']+p['event_input_b'])
                            @ p['event_hidden_w']+p['event_hidden_b'])
        return torch.cat([hidden, gate], dim=1), entities, spatial

    def teacher_heads(self, encoded, teachers, memories, entities, spatial, *, loss_only=False, plan=None):
        from torch_training_inputs import teacher_heads
        return teacher_heads(self, encoded, teachers, memories, entities, spatial, loss_only=loss_only, plan=plan)

    def values(self, hidden):
        p = self.weights
        hidden = hidden[:, :self.config['hidden']]
        return ((hidden @ p['value_hidden_w'] + p['value_hidden_b']).tanh() @ p['value_w'] + p['value_b']).squeeze(-1)
