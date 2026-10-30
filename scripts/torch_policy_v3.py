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

from generals_policy import ARCHITECTURE, MAP_CHANNELS


class _RelationBias(torch.autograd.Function):
    """Avoid the extremely skinny (4 x millions) @ (millions x heads) GEMM.

    Observed geometry is constant, so only a handful of relation coefficients
    need gradients. A parallel reduction is substantially faster on RDNA2.
    Forward arithmetic is unchanged; the fallback input gradient remains general.
    """
    @staticmethod
    def forward(ctx, relative, weight):
        ctx.save_for_backward(relative, weight)
        return relative @ weight

    @staticmethod
    def backward(ctx, gradient):
        relative, weight = ctx.saved_tensors
        grad_relative = gradient @ weight.T if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            dimensions = tuple(range(gradient.ndim-1))
            grad_weight = torch.stack([(relative[..., i, None]*gradient).sum(dimensions)
                                       for i in range(relative.shape[-1])])
        return grad_relative, grad_weight


def relation_bias(relative, weight):
    # Squared map distances exceed FP16 range: geometry stays at learner precision.
    with torch.autocast(relative.device.type, enabled=False):
        if weight.is_cuda and weight.dtype in (torch.float32, torch.float64) and torch.is_grad_enabled():
            return _RelationBias.apply(relative, weight)
        return relative @ weight


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
        values, numeric = encode_values(self, encoded)
        return self.encode_inputs(self.stager.stage(values), numeric)

    def encode_inputs(self, inputs, numeric=None):
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
        entity = torch.tanh(torch.cat([features, types, upgrades], dim=-1) @ p['entity_w'] + p['entity_b'])
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
        q, k, v = [(entity @ p[name]).reshape(batch, count, heads, width).transpose(1, 2)
                   for name in ('attn_q', 'attn_k', 'attn_v')]
        # Geometry follows the learner precision too. Double-precision reference
        # tests retain float64; FP32 training avoids quadratic FP64 temporaries.
        if self.device.type == 'cuda':
            # Construct this O(batch*entities^2) tensor on the accelerator,
            # without a host intermediate or a separate FP64 upload group.
            positions = inputs['positions']
            owner_tensor = inputs['owners']
            delta = positions[:, :, None, :] - positions[:, None, :, :]
            relative = torch.stack([delta[:, :, :, 0], delta[:, :, :, 1], (delta*delta).sum(dim=-1),
                (owner_tensor[:, :, None] == owner_tensor[:, None, :]).to(self.dtype)], dim=-1)
        else:
            positions = numeric[:, :, :2]
            owners = inputs['owners'].numpy()
            delta = positions[:, :, None, :] - positions[:, None, :, :]
            relative = self.array(np.stack([delta[:, :, :, 0], delta[:, :, :, 1], (delta*delta).sum(axis=-1),
                (owners[:, :, None] == owners[:, None, :]).astype(positions.dtype)], axis=-1))
        bias = relation_bias(relative, p['relation_bias']).permute(0, 3, 1, 2)
        scores = q @ k.transpose(-1, -2)/math.sqrt(width) + bias
        scores = scores.masked_fill(~inputs['valid'][:, None, None, :], -torch.inf)
        attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, count, config['entity_hidden'])
        entity = self.normalize(entity + attended @ p['attn_out'])
        entity = self.normalize(entity + torch.tanh(entity @ p['entity_ff']))
        map_size = config['map_side']**2
        scatter = torch.zeros((batch, map_size, 8), dtype=self.dtype, device=self.device)
        locations = inputs['cells']
        real_tensor = inputs['real'][:, :, None]
        scatter = scatter.scatter_add(1, locations[:, :, None].expand(-1, -1, 8),
                                       (entity @ p['scatter_w'])*real_tensor)
        counts = torch.zeros((batch, map_size, 1), dtype=self.dtype, device=self.device).scatter_add(
            1, locations[:, :, None], real_tensor)
        scatter = scatter/counts.clamp_min(1)
        spatial = torch.tanh(inputs['map_inputs']
                             @ p['map_input_w'][:MAP_CHANNELS]
                             + scatter @ p['map_input_w'][MAP_CHANNELS:] + p['map_input_b'])
        side, channels = config['map_side'], config['map_hidden']
        image = spatial.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
        kernel = p['map_conv'].reshape(3, 3, channels, channels).permute(3, 2, 0, 1)
        convolution = F.conv2d(image, kernel, padding=1).permute(0, 2, 3, 1).reshape(batch, map_size, channels)
        spatial = self.normalize(spatial + torch.tanh(convolution + p['map_conv_b']))
        scalar = inputs['scalars'] @ p['scalar_w'] + p['scalar_b']
        for i, prefix in enumerate(('task_command', 'task_validation')):
            scalar = scalar + inputs['task_commands'][:, i] @ p[prefix+'_w']
            embedding = self.type_embedding(inputs['task_command_templates'][:, i])
            scalar = scalar + (embedding @ p[prefix+'_template_w']) * inputs['task_command_template_present'][:, i, None]
        identities = inputs['identities']
        scalar = scalar + (p['base_faction_embedding'][identities[:, 0]]
                           + p['general_embedding'][identities[:, 1]]) @ p['player_identity_w']
        scalar = scalar + inputs['player_flags'] @ p['player_flags_w']
        scalar = torch.tanh(scalar)
        query = scalar @ p['pool_query']
        def pool(mask):
            logits = torch.bmm(entity, query.unsqueeze(-1)).squeeze(-1).masked_fill(~mask, -torch.inf)
            logits = torch.where(mask.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
            weights = logits.softmax(dim=1)*mask
            return torch.bmm(weights.unsqueeze(1), entity).squeeze(1)
        map_scores = torch.bmm(spatial, (scalar @ p['map_pool_query']).unsqueeze(-1)).squeeze(-1)
        map_pool = torch.bmm(map_scores.softmax(dim=1).unsqueeze(1), spatial).squeeze(1)
        features = torch.cat([scalar, pool(inputs['pool_own']), pool(inputs['pool_visible']),
                              pool(inputs['pool_hidden']), pool(inputs['pool_neutral']), map_pool], dim=1)
        return features, entity, spatial

    def events(self, encoded=None, *, prepared=None):
        """Trunk outputs from encoded rows, or from training_preparation's prepared arrays."""
        if prepared is None:
            features, entities, spatial = self.encode_batch(encoded)
        else:
            features, entities, spatial = self.encode_inputs(self.stager.stage(prepared['values']), prepared['numeric'])
        p = self.weights
        hidden = torch.tanh(torch.tanh(features @ p['event_input_w']+p['event_input_b'])
                            @ p['event_hidden_w']+p['event_hidden_b'])
        return hidden, entities, spatial

    def teacher_heads(self, encoded, teachers, memories, entities, spatial, *, loss_only=False, plan=None):
        from torch_training_inputs import teacher_heads
        return teacher_heads(self, encoded, teachers, memories, entities, spatial, loss_only=loss_only, plan=plan)

    def values(self, hidden):
        p = self.weights
        return ((hidden @ p['value_hidden_w'] + p['value_hidden_b']).tanh() @ p['value_w'] + p['value_b']).squeeze(-1)
