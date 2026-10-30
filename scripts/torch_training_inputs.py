"""Typed input buffers and teacher plans; no learned state is cached here."""
from __future__ import annotations

from collections import defaultdict, deque
import math
import os
import numpy as np
import torch

from generals_policy import KINDS
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

    def draw(name, logits, start=None, stop=None):
        mask, choices = inputs[name+'_mask'], inputs[name+'_choices']
        if start is not None:
            mask, choices = mask[start:stop], choices[start:stop]
        return categorical(logits, mask, choices, details=not loss_only,
                           predict=not loss_only or name in ('decision', 'event_delay'))

    logits = memories @ p['verb_w'] + p['verb_b']
    timing, entropy, decision = draw('decision', logits)
    outputs = {'by_head': {'decision': timing}, 'head_counts': {'decision': np.ones(batch, dtype=int)},
               'predictions': [('decision', np.arange(batch), decision)]}
    if not loss_only:
        outputs.update(logp=timing, entropy=entropy, action_kinds=KINDS, token_counts=np.ones(batch, dtype=int),
                       event_prediction=(decision != 0).long(),
                       conditional_prediction=logits[:, 1:].masked_fill(~inputs['conditional_mask'], -torch.inf).argmax(dim=1),
                       order_probability=1-logits.masked_fill(~inputs['decision_mask'], -torch.inf).softmax(dim=1)[:, 0])
    # Command delays use the completed autoregressive argument context below;
    # DEFER has only its verb embedding. This matches deployed sampling.
    delay_context = torch.tanh(memories @ p['context_w']+p['verb_embedding'][inputs['all_verbs']])

    def finish(context):
        lp, ent, prediction = draw('event_delay', context @ p['delay_w']+p['delay_b'])
        outputs['by_head']['delay'] = lp
        outputs['head_counts']['delay'] = np.ones(batch, dtype=int)
        if not loss_only:
            outputs['logp'] = outputs['logp']+lp
            outputs['entropy'] = outputs['entropy']+ent
        outputs['predictions'].append(('delay', np.arange(batch), prediction))
        return outputs

    if not count:
        return finish(delay_context)

    def record(name, values, *, start=None, stop=None):
        lp, ent, prediction = values
        global_rows, ix = plan['categories'][name], inputs[name+'_global']
        if start is not None:
            global_rows, ix = global_rows[start:stop], ix[start:stop]
        previous = outputs['by_head'].get(name, torch.zeros(batch, dtype=model.dtype, device=model.device))
        outputs['by_head'][name] = previous.index_add(0, ix, lp)
        np.add.at(outputs['head_counts'].setdefault(name, np.zeros(batch, dtype=int)), global_rows, 1)
        if not loss_only:
            outputs['logp'] = outputs['logp'].index_add(0, ix, lp)
            outputs['entropy'] = outputs['entropy'].index_add(0, ix, ent)
            np.add.at(outputs['token_counts'], global_rows, 1)
            outputs['predictions'].append((name, global_rows.copy(), prediction))

    command = inputs['command_rows']
    entity, maps = entities[command], spatial[command]
    base = memories[command] @ p['context_w'] + p['verb_embedding'][inputs['verbs']]
    embeddings = entity[inputs['command_range'][:, None], inputs['selected_indices']]
    cumulative = embeddings.cumsum(dim=1)/inputs['selected_divisors'][None, :, None]
    means = torch.cat([torch.zeros((count, 1, config['entity_hidden']), device=model.device,
                                  dtype=model.dtype), cumulative], dim=1)
    for start in range(0, plan['unit_tokens'], 256):
        stop = start+256
        rows, prefixes = inputs['unit_rows'][start:stop], inputs['unit_prefixes'][start:stop]
        context = torch.tanh(base[rows] + means[rows, prefixes] @ p['selected_w'])
        scores = torch.bmm(entity[rows], (context @ p['actor_pointer_w']).unsqueeze(-1)).squeeze(-1)
        unit_logits = torch.cat([scores/math.sqrt(config['entity_hidden']), context @ p['stop_w']+p['stop_b']], dim=1)
        record('unit', draw('unit', unit_logits, start, stop), start=start, stop=stop)
    final = means[inputs['command_range'], inputs['selected_lengths']]
    context = torch.tanh(base + final @ p['selected_w'])

    if len(plan['science_rows']):
        ix, choices = inputs['science_rows'], inputs['science_choices']
        table = p['science_embedding']
        record('science', draw('science', (context[ix] @ p['science_query_w']) @ table.T + p['science_b']))
        context = context.index_copy(0, ix, torch.tanh(context[ix]+table[choices] @ p['science_context_w']))
    if len(plan['command_head_rows']):
        ix, choices = inputs['command_head_rows'], inputs['command_choices']
        table = p['command_embedding']
        record('command', draw('command', (context[ix] @ p['command_query_w']) @ table.T + p['command_b']))
        context = context.index_copy(0, ix, torch.tanh(context[ix]+table[choices] @ p['command_context_w']))

    if len(plan['slot_rows']):
        ix = inputs['slot_rows']
        record('slot', draw('slot', context[ix] @ p['slot_w']+p['slot_b']))
    if len(plan['target_rows']):
        ix = inputs['target_rows']
        target_logits = torch.bmm(entity[ix], (context[ix] @ p['target_pointer_w']).unsqueeze(-1)).squeeze(-1)
        record('target', draw('target', target_logits/math.sqrt(config['entity_hidden'])))
    if len(plan['template_rows']):
        ix, choices = inputs['template_rows'], inputs['template_choices']
        objects = model.object_embedding(model.identity_template_objects)
        record('template', draw('template', (context[ix] @ p['template_query_w']) @ objects.T + p['template_b']))
        context = context.index_copy(0, ix, torch.tanh(context[ix]+objects[choices] @ p['template_context_w']))
    if placement and len(plan['angle_rows']):
        ix = inputs['angle_rows']
        record('angle', draw('angle', context[ix] @ p['angle_w']+p['angle_b']))
        context = context.index_copy(0, ix, torch.tanh(context[ix]+p['build_angle_embedding'][inputs['angle_choices']]))
    if len(plan['spatial_rows']):
        ix = inputs['spatial_rows']
        cells = torch.tanh(maps[ix] + inputs['relative'] @ p['relative_w'])
        if placement and inputs['build_spatial'].numel():
            build = inputs['build_spatial']
            cells = cells.index_copy(0, build, torch.tanh(cells[build]+inputs['build_features'] @ p['build_map_w']))
        cell_logits = torch.bmm(cells, (context[ix] @ p['spatial_query']).unsqueeze(-1)).squeeze(-1)
        record('cell', draw('cell', cell_logits/math.sqrt(config['map_hidden'])))
        chosen = cells[inputs['spatial_range'], inputs['cell_choices']]
        context = context.index_copy(0, ix, torch.tanh(context[ix]+chosen @ p['cell_context_w']))
        record('offset', draw('offset', context[ix] @ p['offset_w']+p['offset_b']))
    if not placement and len(plan['angle_rows']):
        record('angle', draw('angle', context[inputs['angle_rows']] @ p['angle_w']+p['angle_b']))
    return finish(delay_context.index_copy(0, command, context))
