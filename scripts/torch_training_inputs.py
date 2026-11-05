"""Typed input buffers and teacher plans; no learned state is cached here."""
from __future__ import annotations

from collections import defaultdict, deque
import math
import os
import numpy as np
import torch
from torch.nn import functional as F

from generals_policy import ACTION_RESIDUAL_BLOCKS, KINDS
from training_preparation import teacher_plan


class TensorStager:
    """One copy per dtype; Windows HIP uses verified blocking input transfers."""
    def __init__(self, model):
        self.model, self.pending = model, deque()
        windows_hip = os.name == 'nt' and torch.version.hip is not None
        # Matched full passes on Windows/Radeon exposed repeatable numerical
        # corruption with pinned asynchronous uploads. Blocking column passes
        # agree with reference updates. Keep that platform on the safe path.
        self.pin = model.device.type == 'cuda' and not windows_hip
        self.stats = {'batches': 0, 'device_copies': 0, 'host_bytes': 0,
                      'max_inflight_host_bytes': 0, 'pinned': self.pin}
        if model.device.type == 'cuda' and windows_hip:
            self.stats['blocking_reason'] = 'Windows HIP asynchronous input agreement failure'

    def reap(self):
        while self.pending and self.pending[0][0].query():
            self.pending.popleft()

    def stage(self, values):
        self.reap()
        if len(self.pending) >= 8:
            self.pending[0][0].synchronize()
            self.reap()
        groups = defaultdict(list)
        dtypes = {torch.float32: np.float32, torch.float64: np.float64,
                  torch.int64: np.int64, torch.bool: np.bool_}
        # Preparation tags arrays; 'float' follows the learner's precision.
        tags = {'float': self.model.dtype, 'int64': torch.int64, 'bool': torch.bool}
        for name, (value, dtype) in values.items():
            array = np.asarray(value)
            groups[tags.get(dtype, dtype)].append((name, array))
        result, hosts, bytes_used = {}, [], 0
        for dtype, items in groups.items():
            count = sum(array.size for _, array in items)
            if self.pin:
                try:
                    host = torch.empty(count, dtype=dtype, pin_memory=True)
                except RuntimeError as error:
                    if not any(s in str(error).lower() for s in ('pin_memory allocator', 'not support', 'not implemented')):
                        raise
                    self.pin = False
                    self.stats.update(pinned=False, pinning_fallback=str(error))
                    host = torch.from_numpy(np.empty(count, dtype=dtypes[dtype]))
            else:
                host = torch.from_numpy(np.empty(count, dtype=dtypes[dtype]))
            buffer, cursor, descriptors = host.numpy(), 0, []
            for name, array in items:
                buffer[cursor:cursor+array.size] = array.reshape(-1)
                descriptors.append((name, cursor, array.shape, array.size))
                cursor += array.size
            device = host.to(self.model.device, non_blocking=self.pin)
            if self.model.device.type == 'cuda':
                self.stats['device_copies'] += 1
                if self.pin:
                    hosts.append(host)
            size = count*host.element_size()
            bytes_used += size
            for name, offset, shape, length in descriptors:
                result[name] = device.narrow(0, offset, length).reshape(shape)
        self.stats['batches'] += 1
        self.stats['host_bytes'] += bytes_used
        if hosts:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.model.device))
            self.pending.append((event, hosts, bytes_used))
            self.stats['max_inflight_host_bytes'] = max(self.stats['max_inflight_host_bytes'],
                                                       sum(item[2] for item in self.pending))
        return result


def categorical(logits, mask, choices, *, details=True, predict=True):
    """The complete CPU plan checked these masks/choices before device staging."""
    probabilities = logits.masked_fill(~mask, -torch.inf).log_softmax(dim=1)
    entropy = (-(probabilities.exp()*torch.where(mask, probabilities, torch.zeros_like(probabilities))).sum(dim=1)
               if details else None)
    return (probabilities.gather(1, choices[:, None]).squeeze(1), entropy,
            probabilities.argmax(dim=1) if predict else None)


def teacher_heads(model, encoded, teachers, memories, entities, spatial, *, loss_only=False, plan=None):
    """Vectorized argument heads; teacher selections determine autoregressive control."""
    if plan is None:
        plan = teacher_plan(model, encoded, teachers, entities.shape[1])
    inputs = model.stager.stage(plan['values'])
    p, config = model.weights, model.config
    batch, count = plan['batch'], plan['command_count']
    placement = config['placement_mode'] == 'full_mask'

    def draw(name, logits):
        mask, choices = inputs[name+'_mask'], inputs[name+'_choices']
        return categorical(logits, mask, choices, details=not loss_only,
                           predict=not loss_only or name in ('decision', 'event_delay'))

    core, gate = memories[:, :config['hidden']], memories[:, config['hidden']:]
    x = F.relu(core @ p['verb_in_w']+p['verb_in_b'])
    for block in range(ACTION_RESIDUAL_BLOCKS):
        y = F.relu(model.normalize(x @ p[f'verb_res{block}_a']+p[f'verb_res{block}_a_b']))
        x = x+F.relu(model.normalize(y @ p[f'verb_res{block}_b']+p[f'verb_res{block}_b_b']))
    logits = (F.relu(x) @ p['verb_w'])*torch.sigmoid(gate @ p['verb_gate_w']+p['verb_gate_b'])+p['verb_b']
    timing, entropy, decision = draw('decision', logits)
    outputs = {'by_head': {'decision': timing}, 'head_counts': {'decision': np.ones(batch, dtype=int)},
               'predictions': [('decision', np.arange(batch), decision)]}
    if not loss_only:
        outputs.update(logp=timing, entropy=entropy, action_kinds=KINDS, token_counts=np.ones(batch, dtype=int),
                       event_prediction=(decision != 0).long(),
                       conditional_prediction=logits[:, 1:].masked_fill(~inputs['conditional_mask'], -torch.inf).argmax(dim=1),
                       order_probability=1-logits.masked_fill(~inputs['decision_mask'], -torch.inf).softmax(dim=1)[:, 0])
    # The autoregressive embedding, then the delay, which every later argument sees.
    action = F.relu(F.one_hot(inputs['all_verbs'], len(KINDS)).to(model.dtype) @ p['ar_verb_w']+p['ar_verb_b'])
    ar = ((core @ p['ar_core_w'])*torch.sigmoid(gate @ p['ar_core_gate_w']+p['ar_core_gate_b'])
          + (action @ p['ar_verb_out_w'])*torch.sigmoid(gate @ p['ar_verb_gate_w']+p['ar_verb_gate_b']))
    x = F.relu(F.relu(ar @ p['delay_in_w']+p['delay_in_b']) @ p['delay_hidden_w']+p['delay_hidden_b'])
    lp, ent, prediction = draw('event_delay', x @ p['delay_w']+p['delay_b'])
    outputs['by_head']['delay'] = lp
    outputs['head_counts']['delay'] = np.ones(batch, dtype=int)
    if not loss_only:
        outputs['logp'] = outputs['logp']+lp
        outputs['entropy'] = outputs['entropy']+ent
    outputs['predictions'].append(('delay', np.arange(batch), prediction))
    if not count:
        return outputs
    x = F.one_hot(inputs['event_delay_choices'], len(config['delay_ticks'])).to(model.dtype)
    x = F.relu(F.relu(x @ p['delay_embed_in_w']+p['delay_embed_in_b']) @ p['delay_embed_hidden_w']+p['delay_embed_hidden_b'])
    # A censored teacher delay is unknown: later arguments do not see the placeholder.
    ar = ar+(x @ p['delay_embed_out_w']+p['delay_embed_out_b'])*inputs['delay_known'][:, None]

    def record(name, values):
        lp, ent, prediction = values
        global_rows, ix = plan['categories'][name], inputs[name+'_global']
        previous = outputs['by_head'].get(name, torch.zeros(batch, dtype=model.dtype, device=model.device))
        outputs['by_head'][name] = previous.index_add(0, ix, lp)
        np.add.at(outputs['head_counts'].setdefault(name, np.zeros(batch, dtype=int)), global_rows, 1)
        if not loss_only:
            outputs['logp'] = outputs['logp'].index_add(0, ix, lp)
            outputs['entropy'] = outputs['entropy'].index_add(0, ix, ent)
            np.add.at(outputs['token_counts'], global_rows, 1)
            outputs['predictions'].append((name, global_rows.copy(), prediction))

    command = inputs['command_rows']
    keys = entities[command] @ p['entity_key_w']
    func = inputs['func_types'] @ p['func_w']+p['func_b']
    # Teacher selections fix each step's embedding, so the query LSTM runs across them in one pass.
    chosen = keys[inputs['command_range'][:, None], inputs['selected_indices']]/inputs['entity_counts'][:, None, None]
    steps = (chosen @ p['unit_ar_w']+p['unit_ar_b']).cumsum(dim=1)
    steps = ar[command][:, None]+torch.cat([torch.zeros_like(steps[:, :1]), steps], dim=1)
    if plan['unit_tokens']:
        queries = F.relu(steps @ p['unit_in_w']+p['unit_in_b']+func[:, None]) @ p['unit_query_w']+p['unit_query_b']
        # The step count comes from the host-side plan: reading the staged
        # tensor here would synchronise with the GPU in every microbatch.
        length = max(plan['values']['unit_prefixes'][0])+1
        # One fused LSTM over every selection step (cuDNN/MIOpen on a GPU)
        # instead of a dozen kernels per step. The gate layout is PyTorch's
        # (input, forget, cell, output); the single bias goes in b_ih. cuDNN
        # flattens the weights with views, so they must be contiguous.
        zeros = queries.new_zeros((1, count, queries.shape[-1]))
        bias = p['unit_lstm_b']
        weights = [p['unit_lstm_input_w'].T.contiguous(), p['unit_lstm_hidden_w'].T.contiguous()]
        lstm_outputs = torch.lstm(queries[:, :length].contiguous(), (zeros, zeros),
                                  [*weights, bias, torch.zeros_like(bias)],
                                  True, 1, 0.0, torch.is_grad_enabled(), False, True)[0]
        # Every step scored against its command's entities in one product: the
        # same dot products as gathering a key block per token, without the
        # (tokens, entities, width) gather and its scatter in the backward pass.
        logits = torch.cat([torch.bmm(lstm_outputs, keys.transpose(1, 2)),
                            (lstm_outputs @ p['unit_end_key']).unsqueeze(-1)], dim=2)
        record('unit', draw('unit', logits[inputs['unit_rows'], inputs['unit_prefixes']]))
    context = steps[inputs['command_range'], inputs['selected_lengths']]

    if len(plan['science_rows']):
        ix, choices = inputs['science_rows'], inputs['science_choices']
        table = p['science_embedding']
        record('science', draw('science', (context[ix] @ p['science_query_w']) @ table.T + p['science_b']))
        context = context.index_copy(0, ix, context[ix]+table[choices] @ p['science_context_w'])
    if len(plan['command_head_rows']):
        ix, choices = inputs['command_head_rows'], inputs['command_choices']
        table = p['command_embedding']
        record('command', draw('command', (context[ix] @ p['command_query_w']) @ table.T + p['command_b']))
        context = context.index_copy(0, ix, context[ix]+table[choices] @ p['command_context_w'])

    if len(plan['slot_rows']):
        ix = inputs['slot_rows']
        record('slot', draw('slot', context[ix] @ p['slot_w']+p['slot_b']))
    if len(plan['target_rows']):
        ix = inputs['target_rows']
        query = F.relu(context[ix] @ p['target_in_w']+p['target_in_b']+func[ix]) @ p['target_pointer_w']+p['target_pointer_b']
        record('target', draw('target', torch.bmm(keys[ix], query.unsqueeze(-1)).squeeze(-1)))
    if len(plan['template_rows']):
        ix, choices = inputs['template_rows'], inputs['template_choices']
        objects = model.object_embedding(model.identity_template_objects)
        record('template', draw('template', (context[ix] @ p['template_query_w']) @ objects.T + p['template_b']))
        context = context.index_copy(0, ix, context[ix]+objects[choices] @ p['template_context_w'])
    if placement and len(plan['angle_rows']):
        ix = inputs['angle_rows']
        record('angle', draw('angle', context[ix] @ p['angle_w']+p['angle_b']))
        context = context.index_copy(0, ix, context[ix]+p['build_angle_embedding'][inputs['angle_choices']])
    if len(plan['spatial_rows']):
        ix = inputs['spatial_rows']
        build = (inputs['build_spatial'], inputs['build_features']) if placement and inputs['build_spatial'].numel() else None
        cells, logits = model.location(spatial[command[ix]], context[ix], build)
        # One 256x256 distribution, scored as its map cell and then the offset within it.
        record('cell', draw('cell', logits.logsumexp(dim=2)))
        choices, side = inputs['cell_choices'], config['map_side']
        chosen = cells[inputs['spatial_range'], :, torch.div(choices, side, rounding_mode='floor'), choices % side]
        context = context.index_copy(0, ix, context[ix]+chosen @ p['cell_context_w'])
        record('offset', draw('offset', logits[inputs['spatial_range'], choices]))
    if not placement and len(plan['angle_rows']):
        record('angle', draw('angle', context[inputs['angle_rows']] @ p['angle_w']+p['angle_b']))
    return outputs
