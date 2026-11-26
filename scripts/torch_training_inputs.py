"""Typed input buffers and teacher plans; no learned state is cached here."""
from __future__ import annotations

from collections import defaultdict, deque
import math
import os
import numpy as np
import torch
from torch.nn import functional as F

from generals_policy import LOGIT_CAP, SELECTED_UNIT_DIVISOR


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

    def stage(self, values, consumer=None):
        """Device tensors for tagged arrays; `consumer` is the stream that will use them,
        when it is not the stream this runs on (StagedUpdates' upload stream)."""
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
            if consumer is not None:
                # Allocated on the upload stream but read on `consumer`: the
                # allocator must not hand this block out again before that use ends.
                device.record_stream(consumer)
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


class StagedUpdates:
    """Prepared updates with their device inputs already uploaded, one update ahead.

    A background thread takes each update from the source (for PreparedReader,
    the shared-memory copy out of the reader) and stages every microbatch's
    encoder values and teacher plan on its own CUDA stream; the learner's
    stream waits for that upload on the GPU, so the main thread neither copies
    nor waits. On a 5090 these two steps were ~10 ms of a 53 ms update on the
    main thread (9 Nov). Values, order and kernels are unchanged.
    """
    _END = object()

    def __init__(self, source, model, depth=1):
        import queue
        import threading
        self.source, self.model = source, model
        self.cuda = model.device.type == 'cuda'
        self.stager = TensorStager(model)  # Its own pending uploads: the learner's stager is not thread-safe.
        if self.cuda:
            self.device = torch.device('cuda', model.device.index if model.device.index is not None
                                       else torch.cuda.current_device())
            self.upload = torch.cuda.Stream(self.device)
            self.consumer = torch.cuda.current_stream(self.device)
        self.items, self.stopped = queue.Queue(depth), threading.Event()
        self.thread = threading.Thread(target=self._work, name='staged-updates', daemon=True)
        self.thread.start()

    def _stage(self, update):
        for part in update['microbatches']:
            if self.cuda:
                with torch.cuda.device(self.device), torch.cuda.stream(self.upload):
                    part['staged_values'] = self.stager.stage(part['values'], self.consumer)
                    part['plan']['staged'] = self.stager.stage(part['plan']['values'], self.consumer)
                    part['ready'] = torch.cuda.Event()
                    part['ready'].record(self.upload)
            else:
                part['staged_values'] = self.stager.stage(part['values'])
                part['plan']['staged'] = self.stager.stage(part['plan']['values'])

    def _put(self, item):
        import queue
        while not self.stopped.is_set():
            try:
                self.items.put(item, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def _work(self):
        try:
            for update, cursor in self.source:
                if self.stopped.is_set():
                    return
                self._stage(update)
                if not self._put(('item', (update, cursor))):
                    return
            self._put(('end', None))
        except BaseException as error:  # Reaches the consumer, in order.
            self._put(('error', error))

    def __iter__(self):
        return self

    def __next__(self):
        kind, value = self.items.get()
        if kind == 'item':
            return value
        if kind == 'error':
            raise value
        self.items.put((kind, value))  # Stay exhausted.
        raise StopIteration

    def close(self):
        import queue
        self.stopped.set()
        while self.thread.is_alive():
            try:
                self.items.get(timeout=.1)
            except queue.Empty:
                pass
        self.thread.join()
        close = getattr(self.source, 'close', None)
        if close is not None:
            close()


def cap_logits(logits, mask):
    """generals_policy.cap_logits per row: shifted by the best legal logit (with its gradient), soft-capped to (-cap, 0]."""
    best = logits.detach().masked_fill(~mask, -torch.inf).argmax(1, keepdim=True)
    return torch.tanh((logits-logits.gather(1, best))/LOGIT_CAP)*LOGIT_CAP


def categorical(logits, mask, choices, *, details=True, predict=True, capped=True):
    """The complete CPU plan checked these masks/choices before device staging.

    capped=False for the position terms, whose joint map _location already capped."""
    if capped:
        logits = cap_logits(logits, mask)
    probabilities = logits.masked_fill(~mask, -torch.inf).log_softmax(dim=1)
    entropy = (-(probabilities.exp()*torch.where(mask, probabilities, torch.zeros_like(probabilities))).sum(dim=1)
               if details else None)
    return (probabilities.gather(1, choices[:, None]).squeeze(1), entropy,
            probabilities.argmax(dim=1) if predict else None)


def position_errors(logits, cell_mask, cell_choices, offset_choices, config):
    """(expected, mode) pixel distance from the predicted position distribution to the teacher pixel.

    The distribution is the one the cell and offset terms score: legal cells, then every
    offset within a cell. expected = sum_w p(w)*|w-teacher|; mode = |argmax pixel - teacher|.
    """
    n, fine = config['map_side'], config['position_bins']//config['map_side']
    device = logits.device
    joint = logits.float().masked_fill(~cell_mask[:, :, None], -torch.inf).flatten(1).softmax(dim=1)
    cell, offset = torch.arange(n*n, device=device), torch.arange(fine*fine, device=device)
    x = ((cell % n)[:, None]*fine+offset % fine).flatten().float()
    y = (torch.div(cell, n, rounding_mode='floor')[:, None]*fine+torch.div(offset, fine, rounding_mode='floor')).flatten().float()
    tx = ((cell_choices % n)*fine+offset_choices % fine).float()
    ty = (torch.div(cell_choices, n, rounding_mode='floor')*fine+torch.div(offset_choices, fine, rounding_mode='floor')).float()
    distance = torch.hypot(x[None, :]-tx[:, None], y[None, :]-ty[:, None])
    return (joint*distance).sum(dim=1), distance.gather(1, joint.argmax(dim=1, keepdim=True)).squeeze(1)


# Direction metric (head_loss_breakdown): sectors of the circle around the selected units, and the shortest click
# distance (position-bin pixels) for which a direction counts.
DIRECTION_SECTORS, DIRECTION_MIN_PIXELS = 16, 16


def selection_centres(config, pixels, selected, lengths):
    """(rows, 2) x, y in position-bin pixels: the mean of each row's selected units (scatter-pixel centres)."""
    from generals_policy import MAP_POOL, SCATTER_SCALE
    side = config['map_side']*SCATTER_SCALE//MAP_POOL
    scale = config['position_bins']/side
    chosen = pixels.gather(1, selected)
    used = (torch.arange(selected.shape[1], device=pixels.device)[None] < lengths[:, None]).float()
    x, y = ((chosen % side).float()+.5)*scale, (torch.div(chosen, side, rounding_mode='floor').float()+.5)*scale
    count = used.sum(dim=1).clamp_min(1)
    return torch.stack([(x*used).sum(dim=1)/count, (y*used).sum(dim=1)/count], dim=1)


def direction_errors(logits, cell_mask, cell_choices, offset_choices, centres, config, chunk=256):
    """Per spatial row, relative to the selected units' centre: (true distance px, -log p(true direction sector)
    under the predicted position distribution, mode angle error in degrees, log(mode distance / true distance)).
    Direction separates where units are sent from the exact point clicked (the user's movement accuracy)."""
    n, fine = config['map_side'], config['position_bins']//config['map_side']
    device = logits.device
    cell, offset = torch.arange(n*n, device=device), torch.arange(fine*fine, device=device)
    x = ((cell % n)[:, None]*fine+offset % fine).flatten().float()+.5
    y = (torch.div(cell, n, rounding_mode='floor')[:, None]*fine+torch.div(offset, fine, rounding_mode='floor')).flatten().float()+.5
    tx = ((cell_choices % n)*fine+offset_choices % fine).float()+.5
    ty = (torch.div(cell_choices, n, rounding_mode='floor')*fine+torch.div(offset_choices, fine, rounding_mode='floor')).float()+.5
    out = []
    for start in range(0, len(logits), chunk):
        part = slice(start, start+chunk)
        joint = logits[part].float().masked_fill(~cell_mask[part, :, None], -torch.inf).flatten(1).softmax(dim=1)
        cx, cy = centres[part, 0:1], centres[part, 1:2]
        angle = torch.atan2(y[None]-cy, x[None]-cx)
        sector = (((angle+math.pi)/(2*math.pi)*DIRECTION_SECTORS).long()).clamp_max(DIRECTION_SECTORS-1)
        mass = torch.zeros((joint.shape[0], DIRECTION_SECTORS), device=device).scatter_add(1, sector, joint)
        true_angle = torch.atan2(ty[part]-cy[:, 0], tx[part]-cx[:, 0])
        true_sector = (((true_angle+math.pi)/(2*math.pi)*DIRECTION_SECTORS).long()).clamp_max(DIRECTION_SECTORS-1)
        sector_nll = -mass.gather(1, true_sector[:, None])[:, 0].clamp_min(1e-12).log()
        mode = joint.argmax(dim=1)
        mx, my = x[mode], y[mode]
        mode_angle = torch.atan2(my-cy[:, 0], mx-cx[:, 0])
        error = torch.rad2deg(torch.remainder(mode_angle-true_angle+math.pi, 2*math.pi)-math.pi).abs()
        true_distance = torch.hypot(tx[part]-cx[:, 0], ty[part]-cy[:, 0])
        ratio = (torch.hypot(mx-cx[:, 0], my-cy[:, 0]).clamp_min(.5)/true_distance.clamp_min(.5)).log()
        out.append(torch.stack([true_distance, sector_nll, error, ratio], dim=1))
    return torch.cat(out)


def teacher_heads(model, core, bundle, plan, *, loss_only=False, position_metrics=False):
    """Every head's teacher-forced log-probabilities for one microbatch.

    core: (rows, W) core outputs; bundle: the encoder's keys, [Mean, Max] summary and 1x1 map
    for the same rows. The heads are independent given the selected action (SCC), so each
    reads the teacher's action, units and arguments. position_metrics adds
    outputs['position_errors'] (position_errors) and outputs['wait_probability'] (p(wait) per row).
    """
    inputs = plan['staged'] if 'staged' in plan else model.stager.stage(plan['values'])
    p, config = model.weights, model.config
    batch, count = plan['batch'], plan['command_count']
    placement = config['placement_mode'] == 'full_mask'

    def draw(name, logits):
        mask, choices = inputs[name+'_mask'], inputs[name+'_choices']
        return categorical(logits, mask, choices, details=not loss_only,
                           predict=not loss_only or name == 'decision',
                           capped=name not in ('cell', 'offset'))

    actions = inputs['actions']
    logits = model.action_logits(core)
    paced = config.get('network') == 'paced'
    if paced:
        timing, entropy, decision, wait = paced_decision(model, core, logits, inputs, plan, loss_only)
    else:
        timing, entropy, decision = draw('decision', logits)
    from training_preparation import head_counts
    outputs = {'by_head': {'decision': timing}, 'head_counts': head_counts(plan),
               'predictions': [('decision', np.arange(batch), model.action_kinds[decision])]}
    if paced and len(plan['values']['pace_rows'][0]):
        outputs['by_head']['pace'] = pace_loglikelihood(model, core, inputs, batch)
    if (position_metrics or not loss_only) and not paced:
        with torch.no_grad():
            wait = cap_logits(logits, inputs['decision_mask']).masked_fill(
                ~inputs['decision_mask'], -torch.inf).softmax(dim=1)[:, 0]
    if position_metrics:  # Validation: p(wait) per row, for the report's expected wait share.
        outputs['wait_probability'] = wait
    if not loss_only:
        legal = inputs['decision_mask'] & (model.action_kinds != 0)[None]
        outputs.update(logp=timing, entropy=entropy, action_kinds=model.action_kinds,
                       event_prediction=(model.action_kinds[decision] != 0).long(),
                       conditional_prediction=model.action_kinds[logits.masked_fill(~legal, -torch.inf).argmax(dim=1)],
                       order_probability=1-wait)
    if not count:
        return outputs

    def record(name, values):
        lp, ent, prediction = values
        global_rows, ix = plan['categories'][name], inputs[name+'_global']
        previous = outputs['by_head'].get(name, torch.zeros(batch, dtype=lp.dtype, device=lp.device))
        outputs['by_head'][name] = previous.index_add(0, ix, lp)
        if not loss_only:
            outputs['logp'] = outputs['logp'].index_add(0, ix, lp)
            outputs['entropy'] = outputs['entropy'].index_add(0, ix, ent)
            outputs['predictions'].append((name, global_rows.copy(), prediction))

    command = inputs['command_rows']
    core, acts = core[command], actions[command]
    keys, summary = bundle['keys'][command].to(core.dtype), bundle['summary'][command].to(core.dtype)
    if plan['unit_tokens']:
        # Teacher selections fix each step's context, so the query LSTM runs across them in one pass.
        chosen = keys[inputs['command_range'][:, None], inputs['selected_indices']]/SELECTED_UNIT_DIVISOR
        steps = (chosen @ p['unit_ar_w']+p['unit_ar_b']).cumsum(dim=1)
        context = model.unit_context(core, acts)[:, None]
        contexts = context+torch.cat([torch.zeros_like(steps[:, :1]), steps], dim=1)
        # The step count comes from the host-side plan: reading the staged tensor here would
        # synchronise with the GPU in every microbatch.
        length = max(plan['values']['unit_prefixes'][0])+1
        queries = model.unit_query(contexts[:, :length])
        # One fused LSTM over every selection step (cuDNN on a GPU). The gate layout is
        # PyTorch's (input, forget, cell, output); the single bias goes in b_ih. cuDNN
        # flattens the weights with views, so they must be contiguous.
        zeros = queries.new_zeros((1, queries.shape[0], queries.shape[-1]))
        bias = p['unit_lstm_b']
        weights = [p['unit_lstm_input_w'].T.contiguous(), p['unit_lstm_hidden_w'].T.contiguous()]
        hidden = torch.lstm(queries.contiguous(), (zeros, zeros), [*weights, bias, torch.zeros_like(bias)],
                            True, 1, 0.0, torch.is_grad_enabled(), False, True)[0]
        rows, prefixes = inputs['unit_rows'], inputs['unit_prefixes']
        record('unit', draw('unit', model.unit_logits(hidden[rows, prefixes], model.unit_keys(keys)[rows], acts[rows])))
    if len(plan['slot_rows']):
        ix = inputs['slot_rows']
        record('slot', draw('slot', model.slot_logits(core[ix], acts[ix], summary[ix])))
    if len(plan['target_rows']):
        ix = inputs['target_rows']
        record('target', draw('target', model.target_logits(core[ix], acts[ix], summary[ix], keys[ix])))
    if len(plan['angle_rows']):
        ix = inputs['angle_rows']
        record('angle', draw('angle', model.angle_logits(core[ix], acts[ix])))
    if len(plan['spatial_rows']):
        ix = inputs['spatial_rows']
        build = (inputs['build_spatial'], inputs['build_features']) if placement and inputs['build_spatial'].numel() else None
        extra = {}
        if config.get('network') == 'movement':
            from network_alphastar import movement_context
            extra['context'] = movement_context(model, bundle, command[ix], keys[ix], inputs['selected_indices'][ix],
                                                inputs['selected_lengths'][ix])
        logits = model.location(bundle['map'][command[ix]].to(core.dtype), core[ix], acts[ix], summary[ix], build, **extra)
        # One 256x256 distribution, scored as its map cell and then the offset within it.
        cell = draw('cell', logits.logsumexp(dim=2))
        choices = inputs['cell_choices']
        if position_metrics:
            with torch.no_grad():
                outputs['position_errors'] = position_errors(logits.detach(), inputs['cell_mask'], choices,
                                                             inputs['offset_choices'], config)
                if 'pixels' in bundle:  # Direction vs distance around the selected units (head_loss_breakdown).
                    centres = selection_centres(config, bundle['pixels'][command[ix]], inputs['selected_indices'][ix],
                                                inputs['selected_lengths'][ix])
                    outputs['direction'] = direction_errors(logits.detach(), inputs['cell_mask'], choices,
                                                            inputs['offset_choices'], centres, config)
        offset = draw('offset', logits[inputs['spatial_range'], choices])
        record('cell', cell)
        record('offset', offset)
    return outputs


def paced_decision(model, core, logits, inputs, plan, loss_only):
    """network='paced': (log p, entropy, prediction, p(no-op)) of the decision, over the orders (its rows are
    orders only; no-op is never trained). p(no-op) is what the pace head implies for a row of the stored data:
    snapshots at the policy's rate against orders at the row's own rate x exp(pace)."""
    batch, command = plan['batch'], inputs['command_rows']
    mask = inputs['decision_mask'] & (model.action_kinds != 0)[None]
    timing = torch.zeros(batch, dtype=logits.dtype, device=logits.device)
    entropy = torch.zeros_like(timing)
    if plan['command_count']:
        lp, ent, _ = categorical(logits[command], mask[command], inputs['decision_choices'][command], details=not loss_only,
                                 predict=False)
        timing = timing.index_add(0, command, lp)
        if ent is not None:
            entropy = entropy.index_add(0, command, ent)
    with torch.no_grad():
        decision = logits.masked_fill(~mask, -torch.inf).argmax(dim=1)
        orders = inputs['order_rate']*model.pace(core).float().exp()
        snapshots = model.reference.snapshot_rate
        wait = snapshots/(snapshots+orders)
    return timing, entropy, decision, wait


def pace_loglikelihood(model, core, inputs, batch):
    """Per row (zero off the pace rows): the exponential log-likelihood of the delay to the next order, at the
    rate own orders/s x exp(pace): log(rate) - rate x delay when observed, -rate x delay when censored (at least
    that long). The live look acts with the same constant-rate model (generals_policy.decide)."""
    rows = inputs['pace_rows']
    log_rate = inputs['order_rate'][rows].float().log()+model.pace(core[rows]).float()
    loglik = inputs['pace_observed'].float()*log_rate-log_rate.exp()*inputs['pace_delay'].float()
    return torch.zeros(batch, dtype=loglik.dtype, device=loglik.device).index_add(0, rows, loglik)


def head_losses(model, rows, by_head, counts, factors=None):
    """Per-row loss: the action, plus each argument head averaged over its tokens. A no-op has only the action.

    factors: the staged per-row weights (loss_factors), when already on the device."""
    if factors is None:
        factors = model.stager.stage(loss_factors(model.config, rows, counts, by_head))
    loss = -by_head['decision']
    for name, logp in by_head.items():
        if name != 'decision':
            loss = loss-logp*factors[name]
    return loss


def loss_factors(config, rows, counts, names):
    """{head: (per-row weight, 'float')} for head_losses; one staged upload per microbatch."""
    weights = config.get('argument_loss_weights', {})
    factors = {}
    for name in names:
        if name != 'decision':
            factors[name] = weights.get(name, 1)/np.maximum(counts[name], 1), 'float'
    return factors
