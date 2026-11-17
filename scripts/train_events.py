"""Train the memory-free event policy on prepared event shards."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import shutil
import time
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

from event_dataset import EventDataset
from event_store import feature_identity
from event_resources import snapshot, within_budget
from event_training_batches import PreparedReader, prepared as prepared_batches
from generals_policy import ARCHITECTURE, NeuralPolicy
from nn_numpy import Adam
from prepared_io import _digest, _write_json
from training_preparation import MICROBATCH_COST, PreparationSpec, microbatches, prepare_update


PRECISIONS = ('fp32', 'tf32', 'bf16', 'fp16')
# Share of VRAM the caching allocator may reserve on Windows/HIP (see EventLearner); CUDA needs no cap.
WINDOWS_HIP_MEMORY_FRACTION = 0.95

# Game-time buckets (minutes) for validation by phase: late-game coverage is thin.
PHASES = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, None))
# Validation distance-to-click (position_report): order-length bands in pixels from the
# selected units' centre, for the movement kinds; hit radii for the predicted mode.
ORDER_LENGTH_BANDS = (15, 40, 80)
MOVEMENT_KINDS = ('move', 'attack_move', 'add_waypoint')
CLICK_RADII = (3, 10)


# Validation scores that early stopping, LR decay and best.npz can follow: every
# decision, only decisions by players rated 1400+ (by_rating bands), or only
# vanilla-general players' decisions (by_general; three-faction China pilot, 14 Nov
# 2026: the vanilla perspective is what a vanilla-only baseline is compared on).
SELECT_ON = {'all': None, '1400+': ('by_rating', ('1400-1700', '1700+')), 'vanilla': ('by_general', ('vanilla',))}


def selection_score(validation, select_on='all'):
    """Mean validation NLL over the selected decisions; None when there are none."""
    if SELECT_ON[select_on] is None:
        return validation['mean_nll'] if validation.get('decisions') else None
    table, bands = SELECT_ON[select_on]
    parts = [validation[table][name] for name in bands if name in validation.get(table, {})]
    decisions = sum(part['decisions'] for part in parts)
    return sum(part['mean_nll']*part['decisions'] for part in parts)/decisions if decisions else None


def rating_band(rating):
    """The acting player's pre-match Overall rating in run_prepared_training's validation bands."""
    from run_prepared_training import RATING_BANDS, band
    return 'unrated' if rating is None else band(rating, RATING_BANDS)


def stride_order_with_shape_env():
    """Work around a torch 2.14 Inductor bug that kills --compile on real data.

    graph.py's run_node sorts a lowered view's strides with ir.get_stride_order
    but no shape env, so plain sympy compares them. A one-row microbatch saves
    masks such as (slots > 0)[..., None] as [1, s1, 4, 1] with strides
    (4*s1, 4, 1, 1), and sympy cannot decide 1 < s1 ("cannot determine truth
    value of Relational"). lowering.py already passes V.graph.sizevars.shape_env;
    this supplies it at the call sites that omit it, which guards on the
    comparison like every other symbolic stride decision.
    """
    import torch._inductor.ir as ir
    from torch._inductor.virtualized import V
    if getattr(ir.get_stride_order, 'with_shape_env', False):
        return
    original = ir.get_stride_order
    def get_stride_order(seq, shape_env=None):
        if shape_env is None:
            try:
                shape_env = V.graph.sizevars.shape_env
            except (AttributeError, AssertionError):
                pass
        return original(seq, shape_env)
    get_stride_order.with_shape_env = True
    ir.get_stride_order = get_stride_order


# Per-update counters every rank records identically (one optimizer step, one
# norm): a merge keeps rank 0's, and sums the per-row counters of all ranks.
RANK_INVARIANT = ('updates', 'gradient_norm', 'skipped_updates')


class SingleProcess:
    """The collective operations of a one-process run: all of them are identities."""
    rank, world, share = 0, 1, None

    def any(self, *flags):
        return [bool(flag) for flag in flags]

    def merge(self, counts):
        return Counter(counts)

    def same(self, value):
        return value

    def reduce_gradients(self, parameters):
        pass

    def barrier(self):
        pass

    def close(self):
        pass


class DistributedRanks(SingleProcess):
    """Data-parallel ranks on one host (scripts/train_events.py --world-size).

    Every rank reads the same global batches and prepares every world-th row
    (event_training_batches.prepare_share); losses are divided by the global
    row count and gradients summed, so N ranks compute the single-process
    update of the same batch up to floating-point summation order. Control
    decisions (stop, resources, checkpoints) and counters go through a CPU
    gloo group, so they never wait for the GPU.
    """
    def __init__(self, rank, world, address, device_type, backend=None):
        import torch
        import torch.distributed as dist
        self.torch, self.dist = torch, dist
        self.rank, self.world, self.share = rank, world, (rank, world)
        options = {}
        if device_type == 'cuda':
            options['device_id'] = torch.device('cuda', torch.cuda.current_device())
        backend = backend or ('nccl' if device_type == 'cuda' else 'gloo')
        if backend == 'gloo':
            options = {}
        dist.init_process_group(backend, init_method=address, rank=rank, world_size=world, **options)
        self.control = dist.new_group(backend='gloo')

    def any(self, *flags):
        values = self.torch.tensor([int(bool(flag)) for flag in flags], dtype=self.torch.int64)
        self.dist.all_reduce(values, op=self.dist.ReduceOp.MAX, group=self.control)
        return [bool(value) for value in values.tolist()]

    def gather(self, value):
        values = [None]*self.world
        self.dist.all_gather_object(values, value, group=self.control)
        return values

    def merge(self, counts):
        merged = Counter()
        for rank, part in enumerate(self.gather(dict(counts))):
            for key, value in part.items():
                if key not in RANK_INVARIANT:
                    merged[key] += value
                elif rank == 0:
                    merged[key] = value
        return merged

    def same(self, value):
        values = self.gather(value)
        if any(item != values[0] for item in values):
            raise RuntimeError(f'Ranks disagree: {values}')
        return value

    def reduce_gradients(self, parameters):
        """Sum gradients; a parameter no rank used keeps no gradient, as in one process.

        Adam counts steps per parameter, so a zero gradient would not be the same.
        """
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        used = self.any(*(parameter.grad is not None for parameter in parameters))
        gradients = []
        for parameter, needed in zip(parameters, used):
            if needed:
                if parameter.grad is None:
                    parameter.grad = self.torch.zeros_like(parameter)
                gradients.append(parameter.grad)
        if gradients:
            flat = _flatten_dense_tensors(gradients)
            self.dist.all_reduce(flat)
            for gradient, total in zip(gradients, _unflatten_dense_tensors(flat, gradients)):
                gradient.copy_(total)

    def barrier(self):
        self.dist.barrier(group=self.control)

    def close(self):
        if self.dist.is_initialized():
            self.dist.destroy_process_group()


def phase(row):
    minutes = row['frame']/row.get('source_ticks_per_second', 60)/60
    low, high = next(bounds for bounds in PHASES if bounds[1] is None or minutes < bounds[1])
    return f'{low:02d}-{high:02d}' if high is not None else f'{low:02d}+'


# Decisions whose loss exceeds this are written to the run's outliers.jsonl with
# their match, frame and per-head loss terms (reporting only; training is unchanged).
OUTLIER_NLL = 30.0


def _plain(value, limit=16):
    """A JSON-safe, size-bounded copy of a teacher target for outlier reports."""
    if isinstance(value, dict):
        return {str(key): _plain(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        return [_plain(item, limit) for item in items[:limit]]+(['...%d more' % (len(items)-limit)] if len(items) > limit else [])
    if isinstance(value, np.generic):
        return value.item()
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def loss_terms(config, target, by_head, counts):
    """One row's per-head loss terms, as event_losses weights them: {head: term}."""
    weights = config.get('argument_loss_weights', {})
    terms = {}
    for name, logp in by_head.items():
        if name == 'decision':
            factor = 1.0
        elif name == 'delay':
            factor = config.get('delay_loss_weight', .2) if 'delay' in target else 0.0
        else:
            factor = weights.get(name, 1)/max(int(counts[name]), 1)
        if factor and float(logp):
            terms[name] = -float(logp)*factor
    return terms


def event_losses(model, rows, output):
    """Equal decision examples; token averages keep large selections bounded."""
    weights = model.config.get('argument_loss_weights', {})
    factors = {}
    for name in output['by_head']:
        if name == 'delay':
            factors[name] = [model.config.get('delay_loss_weight', .2) if 'delay' in row['target'] else 0 for row in rows]
        elif name != 'decision':
            factors[name] = weights.get(name, 1)/np.maximum(output['head_counts'][name], 1)
    # One staged upload: a pageable copy per head waits for the GPU each time.
    staged = model.stager.stage({name: (value, 'float') for name, value in factors.items()}) if factors else {}
    loss = -output['by_head']['decision']
    for name, logp in output['by_head'].items():
        if name != 'decision':
            loss = loss-logp*staged[name]
    return loss


class EventLearner:
    def __init__(self, reference, optimizer=None, *, device='cpu', threads=1, pair_budget=8_000_000,
                 precision='fp32', compile_encoder=False, ranks=None):
        # fp32: strict IEEE matmul/convolution. tf32: NVIDIA TensorFloat-32 for
        # FP32 matmul/convolution. fp16/bf16: autocast of the trunk with FP32
        # master weights; fp16 adds dynamic loss scaling.
        import torch
        from torch_policy_v3 import TorchPolicyV3
        if (reference.config.get('architecture') != ARCHITECTURE or threads < 1 or pair_budget < 1
                or precision not in PRECISIONS or (precision != 'fp32' and device != 'cuda')):
            raise ValueError('Invalid event learner configuration')
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision('high' if precision == 'tf32' else 'highest')
        if device == 'cuda':
            torch.backends.cudnn.conv.fp32_precision = 'tf32' if precision == 'tf32' else 'ieee'
            # The unit head's fused LSTM runs through cuDNN, whose RNN default is TF32.
            torch.backends.cudnn.rnn.fp32_precision = 'tf32' if precision == 'tf32' else 'ieee'
            if os.name == 'nt' and torch.version.hip is not None:
                # Windows/HIP grants allocations past VRAM from system RAM instead of failing, so the caching
                # allocator never frees and retries as it does on CUDA: on the 12 GB Radeon it spilled up to 11 GB
                # (16 Nov 2026) while tensors peaked at 9.3 GB. A cap restores CUDA's free-and-retry behaviour.
                torch.cuda.set_per_process_memory_fraction(WINDOWS_HIP_MEMORY_FRACTION)
        self.torch, self.reference = torch, reference
        self.ranks = ranks or SingleProcess()
        self.outliers = []  # Extreme-loss decisions since train() last wrote them.
        self.model = TorchPolicyV3(reference, device=device, dtype=torch.float32)
        self.parameter_list = list(self.model.parameters())
        if compile_encoder:
            # Fuses the entity/map encoder, where most launched ops are (+12-25%
            # on an RTX 5070 Ti); heads stay eager. Dynamic shapes: every
            # microbatch has its own batch and entity counts.
            import importlib.util
            if device != 'cuda' or torch.version.hip is not None or importlib.util.find_spec('triton') is None:
                raise ValueError('--compile needs an NVIDIA CUDA device with Triton')
            stride_order_with_shape_env()
            self.model.encode_inputs = torch.compile(self.model.encode_inputs, dynamic=True)
            # The heads' fixed-structure trunk too: the rest of the heads branch on the plan.
            from torch_training_inputs import action_trunk
            self.model.action_trunk = torch.compile(action_trunk, dynamic=True)
        self.pair_budget = pair_budget
        # FP16 autocast keeps FP32 master weights and Adam state; dynamic loss
        # scaling skips updates whose scaled gradients overflow.
        self.fp16 = precision == 'fp16'
        self.autocast = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(precision)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.fp16)
        self.mirror = optimizer or Adam(reference.parameters, 3e-4)
        # foreach, not fused: fused Adam broke bitwise checkpoint continuation on
        # Radeon (3 of 12 runs), for ~0.5 ms of a 150 ms update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mirror.lr, foreach=True)
        self.step_count = self.mirror.step_count
        if self.step_count:
            steps = getattr(self.mirror, 'parameter_steps', None)
            if steps is None or set(steps) != set(self.model.weights):
                raise ValueError('Event continuation needs per-parameter Adam counters')
            for name, parameter in self.model.weights.items():
                if not steps[name]:
                    continue
                self.optimizer.state[parameter] = {
                    'step': torch.tensor(float(steps[name])),
                    'exp_avg': torch.tensor(self.mirror.m[name], device=self.model.device, dtype=self.model.dtype),
                    'exp_avg_sq': torch.tensor(self.mirror.v[name], device=self.model.device, dtype=self.model.dtype)}

    def staged(self, stream):
        """`stream`'s updates with their inputs uploaded one update ahead, off the main thread."""
        from torch_training_inputs import StagedUpdates
        return StagedUpdates(stream, self.model)

    def batch(self, rows, *, train):
        """Prepare in-process, then run one update (or a validation pass) over rows."""
        cached = rows[0]['encoded'].get('_training_legality')
        self.model.legality_key = cached['key'] if cached else None
        return self.run(prepare_update(self.model, rows, self.pair_budget), train=train)

    def run(self, prepared, *, train):
        """One update from training_preparation.prepare_update's arrays."""
        torch, model = self.torch, self.model
        totals = Counter()
        # Predictions and losses stay on the device until the update is queued,
        # then come back in one transfer: no synchronisation per microbatch.
        decisions, delays, losses_by_part, heads_by_part = [], [], [], []
        position_errors = []  # Validation only: (part, expected, mode) distance-to-click.
        wait_sums = []  # Validation only: summed p(wait), the expected number of waits.
        # A rank's share of a global batch is divided by the whole batch's rows.
        rows_total = prepared.get('global_rows', prepared['rows'])
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            for part in prepared['microbatches']:
                rows = part['rows']
                if 'ready' in part:  # Uploaded by StagedUpdates: a GPU-side wait, not a CPU one.
                    torch.cuda.current_stream(model.device).wait_event(part['ready'])
                # FP16 covers the trunk, where nearly all the work is; the small
                # argument heads and losses stay FP32.
                with torch.autocast('cuda', dtype=self.autocast or torch.float16, enabled=self.autocast is not None):
                    trunk = model.events(prepared=part)
                hidden, entities, spatial = (value.float() for value in trunk)
                del trunk
                output = model.teacher_heads(None, [row['target'] for row in rows], hidden,
                                             entities, spatial, loss_only=True, plan=part['plan'],
                                             position_metrics=not train)
                if 'position_errors' in output:
                    position_errors.append((part, *output['position_errors']))
                if 'wait_probability' in output:
                    wait_sums.append(output['wait_probability'].sum())
                losses = event_losses(model, rows, output)
                if train:
                    self.scaler.scale(losses.sum()/rows_total).backward()
                decisions.append(output['predictions'][0][2].detach())
                delays.append(next(item[2] for item in output['predictions'] if item[0] == 'delay').detach())
                losses_by_part.append(losses.detach())
                heads_by_part.append(({name: value.detach() for name, value in output['by_head'].items()},
                                      output['head_counts']))
                # Release the preceding microbatch's graph before allocating
                # the next one's tensors.
                del hidden, entities, spatial, output, losses
            norm = None
            if train:
                if self.ranks.rank == 0:  # Once per update, not once per rank.
                    regularization = model.regularization_loss()
                    if regularization.requires_grad:
                        self.scaler.scale(regularization).backward()
                self.ranks.reduce_gradients(self.parameter_list)
                self.scaler.unscale_(self.optimizer)
                # Under loss scaling an overflow is expected occasionally: the
                # scaler skips that step and lowers its scale instead.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=not self.fp16,
                                                      foreach=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step_count += 1
                totals['updates'] = 1
        if losses_by_part:
            guesses = torch.cat([torch.stack([d, e]) for d, e in zip(decisions, delays)], dim=1).cpu().numpy()
            # After the rows: the gradient norm.
            values = torch.cat(losses_by_part+([norm.reshape(1).to(losses_by_part[0].dtype)] if norm is not None else []))
        else:  # This rank's share of a small final batch was empty; it still joined the update.
            guesses = np.zeros((2, 0), dtype=np.int64)
            values = norm.reshape(1).float() if norm is not None else torch.zeros(0)
        values = values.cpu().numpy()
        start = 0
        for index, (part, row_losses) in enumerate(zip(prepared['microbatches'], np.split(values, np.cumsum([len(p['rows']) for p in
                                                                                       prepared['microbatches']])))):
            rows = part['rows']
            self._count(totals, rows, guesses[0, start:start+len(rows)], guesses[1, start:start+len(rows)], row_losses)
            self._outliers(totals, part, row_losses, heads_by_part[index], train)
            start += len(rows)
            totals['microbatches'] += 1
            totals['entity_pair_work'] += len(rows)*part['entity_count']**2
        if wait_sums:
            totals['expected_defers'] += float(torch.stack(wait_sums).sum())
        if position_errors:  # One transfer for the whole update.
            sizes = [len(expected) for _, expected, _ in position_errors]
            errors = torch.stack([torch.cat([e for _, e, _ in position_errors]),
                                  torch.cat([m for _, _, m in position_errors])]).cpu().numpy()
            for (part, _, _), expected, mode in zip(position_errors, *(np.split(e, np.cumsum(sizes)[:-1]) for e in errors)):
                position_report(totals, part, expected, mode)
        if norm is not None:
            norm = float(values[-1])
            if math.isfinite(norm):
                totals['gradient_norm'] = norm
            else:
                totals['skipped_updates'] = 1
        return totals

    def _recheck(self, part, extreme):
        """The extreme rows' losses recomputed now from the host's copy of the inputs.

        'unstaged' repeats the run's precision without StagedUpdates' uploads; 'fp32'
        drops autocast too. Equal to the recorded loss: the data. Normal again
        unstaged: the staged inputs. Normal only in fp32: reduced precision. A training
        update has already been applied, so small differences are expected there.
        """
        torch, model = self.torch, self.model
        plain = {key: value for key, value in part.items() if key not in ('staged_values', 'ready')}
        plain['plan'] = {key: value for key, value in part['plan'].items() if key != 'staged'}
        rows, results = part['rows'], {}
        with torch.no_grad():
            for name, dtype in (('unstaged', self.autocast), ('fp32', None)):
                with torch.autocast('cuda', dtype=dtype or torch.float16, enabled=dtype is not None):
                    trunk = model.events(prepared=plain)
                hidden, entities, spatial = (value.float() for value in trunk)
                output = model.teacher_heads(None, [row['target'] for row in rows], hidden, entities, spatial,
                                             loss_only=True, plan=plain['plan'])
                losses = event_losses(model, rows, output)
                results[name] = losses[torch.as_tensor(extreme, device=losses.device)].float().cpu().numpy()
        return results

    def _outliers(self, totals, part, row_losses, heads, train):
        """Record decisions above OUTLIER_NLL; their head terms leave the device only then."""
        rows = part['rows']
        extreme = np.flatnonzero(~(row_losses[:len(rows)] <= OUTLIER_NLL))  # NaN counts as extreme
        if not len(extreme):
            return
        try:
            recheck = self._recheck(part, extreme)
        except Exception as error:  # noqa: BLE001 - a report must never stop training
            recheck = {'error': f'{type(error).__name__}: {error}'}
        by_head, counts = heads
        values = {name: value[extreme].float().cpu().numpy() for name, value in by_head.items()}
        for slot, row in enumerate(extreme):
            source = rows[row]
            frame = source.get('frame')
            terms = loss_terms(self.model.config, source['target'], {name: value[slot] for name, value in values.items()},
                               {name: count[row] for name, count in counts.items()})
            self.outliers.append({'split': 'train' if train else 'validation', 'update': self.step_count,
                                  'match_key': source.get('match_key'), 'player': source.get('player'), 'frame': frame,
                                  'minute': None if frame is None else frame/source.get('source_ticks_per_second', 60)/60,
                                  'rating': source.get('rating'), 'nll': float(row_losses[row]),
                                  'terms': terms, 'target': _plain(source['target']),
                                  'recheck': {name: (float(value[slot]) if not isinstance(value, str) else value)
                                              for name, value in recheck.items()},
                                  'after_update': bool(train)})
            totals['outliers'] += 1
            totals['outlier_loss_sum'] += float(row_losses[row])

    def _count(self, totals, rows, predicted, delay_predictions, row_losses):
        """Per-row decision/delay statistics of one microbatch, vectorised; same sums and order."""
        kinds = self.model.reference.kinds
        targets = [row['target'] for row in rows]
        labels = np.fromiter((kinds.index(t['kind']) for t in targets), dtype=np.int64, count=len(rows))
        hits = predicted == labels
        totals['loss_sum'] += float(row_losses.sum())
        totals['decisions'] += len(rows)
        totals['correct'] += int(hits.sum())
        # Per rating band: whether the scarce high-rated players' decisions improve, not just the average.
        for band, loss, hit in zip((rating_band(row.get('rating')) for row in rows), row_losses, hits):
            totals['rating:'+band+':decisions'] += 1
            totals['rating:'+band+':loss_sum'] += float(loss)
            totals['rating:'+band+':correct'] += int(hit)
        # Per acting general: in mixed China matches both players are supervised.
        for general, loss, hit in zip((row.get('general', '<unknown>') for row in rows), row_losses, hits):
            totals['general:'+general+':decisions'] += 1
            totals['general:'+general+':loss_sum'] += float(loss)
            totals['general:'+general+':correct'] += int(hit)
        # Strategy input z per row (event_dataset): given, blanked, none (under six buildings), after (window over).
        # The comparable loss is the blanked/none/after rows; given vs blanked in the opening shows z's use.
        for status, loss, hit in zip((row.get('opener') or 'unknown' for row in rows), row_losses, hits):
            totals['opener:'+status+':decisions'] += 1
            totals['opener:'+status+':loss_sum'] += float(loss)
            totals['opener:'+status+':correct'] += int(hit)
        timed = np.fromiter((row['frame'] is not None for row in rows), dtype=bool, count=len(rows))
        if timed.any():  # prepared shards record each event's source frame
            minutes = np.array([row['frame']/row.get('source_ticks_per_second', 60)/60
                                for row, keep in zip(rows, timed) if keep])
            buckets = np.searchsorted([bounds[1] for bounds in PHASES[:-1]], minutes, side='right')
            names = [f'{low:02d}-{high:02d}' if high is not None else f'{low:02d}+' for low, high in PHASES]
            # bincount adds in row order, as the per-row loop did.
            loss_sums = np.bincount(buckets, weights=row_losses[timed].astype(np.float64), minlength=len(PHASES))
            counts = np.bincount(buckets, minlength=len(PHASES))
            correct = np.bincount(buckets, weights=hits[timed], minlength=len(PHASES))
            for index in np.flatnonzero(counts):
                bucket = 'minute:'+names[index]+':'
                totals[bucket+'decisions'] += int(counts[index])
                totals[bucket+'loss_sum'] += float(loss_sums[index])
                totals[bucket+'correct'] += int(correct[index])
        for category, members in (('defers', labels == 0), ('commands', labels != 0)):
            if members.any():  # A category's counters exist once it has a row.
                totals[category] += int(members.sum())
                totals[category+'_correct'] += int(hits[members].sum())
        totals.update('kind:'+t['kind'] for t in targets)
        delayed = [i for i, t in enumerate(targets) if 'delay' in t]
        totals['delay_labels'] += len(delayed)
        if delayed:
            ticks = np.asarray(self.model.config['delay_ticks'])
            guess = delay_predictions[delayed].astype(np.int64)
            label = np.array([targets[i]['delay'] for i in delayed])
            totals['delay_abs_ticks'] += int(np.abs(ticks[guess]-ticks[label]).sum())
            totals['delay_correct'] += int((guess == label).sum())

    @property
    def lr(self):
        return self.mirror.lr

    def set_lr(self, lr):
        # The mirror's rate is what checkpoints save and resumes restore.
        self.mirror.lr = lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def save(self, path, metadata):
        def host(tensor):
            return tensor.detach().cpu().numpy()
        weights, first, second, steps = {}, {}, {}, {}
        for name, parameter in self.model.weights.items():
            weights[name] = host(parameter)
            state = self.optimizer.state.get(parameter)
            steps[name] = int(state['step'].item()) if state else 0
            # A parameter Adam has not stepped has no state yet: its moments are zero.
            first[name] = host(state['exp_avg']) if state else np.zeros_like(weights[name])
            second[name] = host(state['exp_avg_sq']) if state else np.zeros_like(weights[name])
        rng = {'torch': self.torch.get_rng_state().tolist(),
               'cuda': [state.tolist() for state in self.torch.cuda.get_rng_state_all()] if self.torch.cuda.is_available() else [],
               'python': random.getstate(),
               'numpy': [item.tolist() if isinstance(item, np.ndarray) else item for item in np.random.get_state()]}
        if self.fp16:
            metadata = {**metadata, 'event_grad_scaler': self.scaler.state_dict()}
        document = {'config': self.reference.config, 'optimizer_step': self.step_count, 'optimizer_lr': self.lr,
                    'metadata': {**metadata, 'event_rng': rng, 'event_optimizer_steps': steps}}
        write_checkpoint(path, document, weights, first, second)

    def restore_scaler(self, metadata):
        if self.fp16:
            self.scaler.load_state_dict(metadata['event_grad_scaler'])

    def restore_rng(self, metadata):
        rng = metadata['event_rng']
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value
        random.setstate(tuples(rng['python']))
        state = rng['numpy']
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        self.torch.set_rng_state(self.torch.tensor(rng['torch'], dtype=self.torch.uint8))
        if rng['cuda'] and self.model.device.type == 'cuda':
            self.torch.cuda.set_rng_state_all([self.torch.tensor(state, dtype=self.torch.uint8) for state in rng['cuda']])


def order_length_band(distance):
    if distance is None:
        return 'no_units'
    low = 0
    for high in ORDER_LENGTH_BANDS:
        if distance < high:
            return f'{low}-{high}'
        low = high
    return f'{low}+'


def position_report(totals, part, expected, mode):
    """Distance-to-click counters for one microbatch's position rows (pixels; ~10 world units each).

    expected: the predicted distribution's mean distance to the human's pixel; mode: the
    most likely pixel's distance. Teacher-forced: the kind, units and earlier arguments are
    the human's, so this scores where the model would click for the same order.
    """
    plan, rows = part['plan'], part['rows']
    kinds = [rows[plan['command_rows'][i]]['target']['kind'] for i in plan['spatial_rows']]
    for kind, length, e, m in zip(kinds, plan['order_distances'], expected, mode):
        groups = ['position:'+kind+':']
        if kind in MOVEMENT_KINDS:
            groups.append('position_length:'+order_length_band(length)+':')
        for group in groups:
            totals[group+'orders'] += 1
            totals[group+'expected_sum'] += float(e)
            totals[group+'mode_sum'] += float(m)
            for radius in CLICK_RADII:
                totals[group+f'mode_within_{radius}'] += int(m <= radius)


def position_summary(counts, prefix):
    names = sorted({key.split(':')[1] for key in counts if key.startswith(prefix+':')})
    result = {}
    for name in names:
        key = prefix+':'+name+':'
        orders = max(1, counts[key+'orders'])
        result[name] = {'orders': counts[key+'orders'], 'mean_expected_pixels': counts[key+'expected_sum']/orders,
                        'mean_mode_pixels': counts[key+'mode_sum']/orders,
                        **{f'mode_within_{radius}': counts[key+f'mode_within_{radius}']/orders for radius in CLICK_RADII}}
    return result


def summarize(counts):
    phases = sorted({key.split(':')[1] for key in counts if key.startswith('minute:')})
    by_minute = {name: {'decisions': counts['minute:'+name+':decisions'],
                        'mean_nll': counts['minute:'+name+':loss_sum']/max(1, counts['minute:'+name+':decisions']),
                        'decision_accuracy': counts['minute:'+name+':correct']/max(1, counts['minute:'+name+':decisions'])}
                 for name in phases}
    bands = sorted({key.split(':')[1] for key in counts if key.startswith('rating:')})
    by_rating = {name: {'decisions': counts['rating:'+name+':decisions'],
                        'mean_nll': counts['rating:'+name+':loss_sum']/max(1, counts['rating:'+name+':decisions']),
                        'decision_accuracy': counts['rating:'+name+':correct']/max(1, counts['rating:'+name+':decisions'])}
                 for name in bands}
    generals = sorted({key.split(':')[1] for key in counts if key.startswith('general:')})
    by_general = {name: {'decisions': counts['general:'+name+':decisions'],
                         'mean_nll': counts['general:'+name+':loss_sum']/max(1, counts['general:'+name+':decisions']),
                         'decision_accuracy': counts['general:'+name+':correct']/max(1, counts['general:'+name+':decisions'])}
                  for name in generals}
    statuses = sorted({key.split(':')[1] for key in counts if key.startswith('opener:')})
    by_opener = {name: {'decisions': counts['opener:'+name+':decisions'],
                        'mean_nll': counts['opener:'+name+':loss_sum']/max(1, counts['opener:'+name+':decisions']),
                        'decision_accuracy': counts['opener:'+name+':correct']/max(1, counts['opener:'+name+':decisions'])}
                 for name in statuses}
    positions = {}
    if any(key.startswith('position:') for key in counts):
        positions = {'position_error': {'by_kind': position_summary(counts, 'position'),
                                        'movement_by_order_length': position_summary(counts, 'position_length')}}
    return {**{k: v for k, v in counts.items() if not k.startswith(('minute:', 'rating:', 'general:', 'opener:', 'position:', 'position_length:'))},
            'by_minute': by_minute, 'by_rating': by_rating, 'by_general': by_general, 'by_opener': by_opener, **positions,
            'mean_nll': counts['loss_sum']/max(1, counts['decisions']),
            'decision_accuracy': counts['correct']/max(1, counts['decisions']),
            # Validation: the model's expected wait share next to the humans' (calibration of acting vs waiting).
            **({'wait_share': counts['defers']/max(1, counts['decisions']),
                'expected_wait_share': counts['expected_defers']/max(1, counts['decisions'])}
               if 'expected_defers' in counts else {}),
            'command_accuracy': counts['commands_correct']/max(1, counts['commands']),
            'delay_mae_ticks': counts['delay_abs_ticks']/max(1, counts['delay_labels'])}


class EarlyStopping:
    def __init__(self,patience=0,min_delta=0.0,state=None):
        if type(patience) is not int or patience<0 or not math.isfinite(min_delta) or min_delta<0:
            raise ValueError('Invalid early-stopping settings')
        self.patience,self.min_delta=patience,min_delta
        self.best_score,self.best_epoch,self.bad_epochs=None,0,0
        if state:
            if state['patience']!=patience or state['min_delta']!=min_delta:
                raise ValueError('Early-stopping settings changed during resume')
            self.best_score,self.best_epoch,self.bad_epochs=(state[k] for k in ('best_score','best_epoch','bad_epochs'))

    def observe(self,epoch,score):
        if not math.isfinite(score):raise ValueError('Nonfinite validation loss')
        improved=self.best_score is None or score<self.best_score-self.min_delta
        if improved:self.best_score,self.best_epoch,self.bad_epochs=score,epoch,0
        else:self.bad_epochs+=1
        return improved

    @property
    def stopped(self):return bool(self.patience and self.bad_epochs>=self.patience)

    def state(self):
        return dict(patience=self.patience,min_delta=self.min_delta,best_score=self.best_score,
                    best_epoch=self.best_epoch,bad_epochs=self.bad_epochs)


class LearningRate:
    """The learner's rate at every update.

    Three independent parts, multiplied:
    - warmup: a linear ramp over the first `warmup` updates (Adam's second moments are
      empty then, and the clipped first steps are the least reliable of the run);
    - schedule: 'constant', or 'cosine' from the peak to min_lr over `horizon` updates,
      for runs with a planned length;
    - plateau: `decay` each time `patience` consecutive validation rounds fail to improve
      on the best score, then `cooldown` rounds that are not counted. One bad round is
      usually a loss spike, not a plateau (v10 round 3, 11 Nov 2026), so patience
      defaults to 2; an improving round clears the count.
    min_lr floors the decayed rate, not the warmup ramp. The rate is a pure function of
    the update count and state(), so a resume continues it exactly.
    """
    SCHEDULES=('constant','cosine')

    def __init__(self,peak,*,warmup=0,schedule='constant',horizon=0,min_lr=0.0,decay=1.0,patience=2,cooldown=0,state=None):
        if (not math.isfinite(peak) or peak<=0 or schedule not in self.SCHEDULES or not 0<=min_lr<=peak or not 0<decay<=1
                or any(type(v) is not int or v<0 for v in (warmup,horizon,cooldown)) or type(patience) is not int or patience<1
                or (schedule=='cosine' and horizon<1)):
            raise ValueError('Invalid learning-rate schedule')
        self.peak,self.warmup,self.schedule,self.horizon=peak,warmup,schedule,horizon
        self.min_lr,self.decay,self.patience,self.cooldown=min_lr,decay,patience,cooldown
        self.scale,self.bad_rounds,self.cooling=1.0,0,0
        if state:
            self.scale,self.bad_rounds,self.cooling=(state[k] for k in ('scale','bad_rounds','cooling'))

    def settings(self):
        """Everything that fixes the rate; None for a plain constant rate."""
        parts={}
        if self.warmup:parts['warmup']=self.warmup
        if self.schedule!='constant':parts.update(schedule=self.schedule,horizon=self.horizon)
        if self.decay!=1.0:parts.update(decay=self.decay,patience=self.patience,cooldown=self.cooldown)
        if parts:parts['min_lr']=self.min_lr
        return parts or None

    def at(self,step):
        """The rate of the update after `step` completed ones."""
        rate=self.peak
        if self.schedule=='cosine':
            rate=self.min_lr+(self.peak-self.min_lr)*.5*(1+math.cos(math.pi*min(1.0,step/self.horizon)))
        rate=max(self.min_lr,rate*self.scale)
        return rate*min(1.0,(step+1)/self.warmup) if self.warmup else rate

    def observe(self,improved):
        """One validation round; True when the plateau multiplier dropped."""
        if self.decay==1.0:return False
        if improved:
            self.bad_rounds=0
            self.cooling=max(0,self.cooling-1)
            return False
        if self.cooling:
            self.cooling-=1
            return False
        self.bad_rounds+=1
        if self.bad_rounds<self.patience:return False
        self.scale*=self.decay
        self.bad_rounds,self.cooling=0,self.cooldown
        return True

    def state(self):
        return dict(scale=self.scale,bad_rounds=self.bad_rounds,cooling=self.cooling)


def updates_per_epoch(dataset,seed,batch_size,buffer_bytes):
    """Training updates in epoch 0 from the shard index alone; later epochs differ by a few (window boundaries)."""
    return sum(math.ceil(sum(reference[2]['rows'] for reference in window)/batch_size)
               for window in dataset.windows('train',seed=seed,epoch=0,buffer_bytes=buffer_bytes))


def batch_fill(dataset,seed,batch_size,buffer_bytes):
    """Mean rows per training update divided by the requested batch size (epoch 0, from the shard index).

    A batch is cut from one shuffle window (event_dataset.window_batches), and a window's
    last batch is whatever rows remain. When the window holds only about one batch, the
    requested size is never reached or every full batch is followed by a small remainder:
    v10 asked for 512 and averaged 292 rows per update at a 64 MiB window. Near 1 means
    the window holds many batches; raise --buffer-mib until it does.
    """
    rows=sum(reference[2]['rows'] for window in dataset.windows('train',seed=seed,epoch=0,buffer_bytes=buffer_bytes)
             for reference in window)
    return rows/max(1,updates_per_epoch(dataset,seed,batch_size,buffer_bytes))/batch_size


def write_checkpoint(path, document, weights, first, second):
    """Write learner state as NeuralPolicy.load reads it: float32 as trained, uncompressed.

    The hashes are NeuralPolicy.weight_hash/identity_hash, which load verifies
    over the float64 values it widens these to. At 53M parameters this takes
    0.6 s, against 25 s for float64 with zlib; a writer thread was measured and
    cost the training thread more than the write it took over.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for name in sorted(weights):
        digest.update(name.encode()); digest.update(np.ascontiguousarray(weights[name], dtype=np.float64))
    identity = hashlib.sha256(json.dumps(document['config'], sort_keys=True, separators=(',', ':')).encode())
    identity.update(digest.hexdigest().encode())
    document = {**document, 'weight_hash': digest.hexdigest(), 'policy_identity_hash': identity.hexdigest()}
    arrays = {'metadata': np.asarray(json.dumps(document, allow_nan=False))}
    for prefix, values in (('p_', weights), ('m_', first), ('v_', second)):
        arrays.update({prefix+name: value for name, value in values.items()})
    temporary = path.with_name(path.name+'.part')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def copy_checkpoint(source,target):
    temporary=target.with_name(target.name+'.part')
    shutil.copyfile(source,temporary)
    os.replace(temporary,target)


def restore_best_checkpoint(resume,best,stopper,metadata):
    if not stopper.best_epoch:return
    if best.exists():
        with np.load(best,allow_pickle=False) as archive:
            saved=json.loads(str(archive['metadata']))['metadata']
        state=saved.get('event_early_stopping',{})
        if (saved.get('event_dataset_sha256')==metadata['event_dataset_sha256']
                and state.get('best_epoch')==stopper.best_epoch and state.get('best_score')==stopper.best_score):return
    # Recover a crash between publishing latest.npz and its best.npz copy:
    # latest is still the checkpoint written right after the best round.
    rounds=metadata.get('event_history',[])
    if len(rounds)==stopper.best_epoch and rounds[-1].get('cursor')==metadata['event_cursor']:
        copy_checkpoint(Path(resume),best)
    else:raise ValueError('Best checkpoint is missing/stale and cannot be recovered from this resume point')


def validation_marks(dataset,seed,epoch,buffer_bytes,validations):
    """Shuffle windows before whose first batch the epoch's mid-epoch validations run."""
    if validations==1:return []
    count=len(dataset.windows('train',seed=seed,epoch=epoch,buffer_bytes=buffer_bytes))
    return sorted({math.ceil(k*count/validations) for k in range(1,validations)}-{0,count})


def train(args, ranks=None):
    """One run, or one rank of a data-parallel run (ranks: DistributedRanks; rank 0 writes every file)."""
    invocation_start = time.monotonic()
    ranks = ranks or SingleProcess()
    leader = ranks.rank == 0
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    guard = {'minimum_ram': args.min_free_ram_mib*1024**2, 'maximum_private': args.max_private_mib*1024**2,
             'minimum_disk': getattr(args, 'min_free_disk_gib', 80)*1024**3,
             'maximum_gpu': getattr(args, 'max_gpu_mib', 9216)*1024**2}
    if min(guard.values()) < 1:
        raise ValueError('Positive resource budgets required')
    if not within_budget(snapshot(output), **guard):
        raise RuntimeError('Event training is waiting for its RAM/disk resource reserve; rerun to resume')
    import torch
    dataset = EventDataset(args.dataset)
    latest = output/'latest.npz'
    pending = output/'latest.npz.next'
    best = output/'best.npz'
    if (latest.exists() or best.exists()) and not args.resume:
        raise ValueError('Output already has a checkpoint; use --resume explicitly')
    settings = {'seed': args.seed, 'batch_size': args.batch_size, 'buffer_bytes': args.buffer_mib*1024*1024,
                'pair_budget': args.pair_budget, 'device': args.device, 'threads': args.threads, 'lr': args.lr,
                'prefetch_batches': getattr(args, 'prefetch_batches', 1),
                'reader_processes': getattr(args, 'reader_processes', 1),
                'precision': getattr(args, 'precision', 'fp32'), 'compile': getattr(args, 'compile', False),
                'world_size': ranks.world}
    # Validation rounds per epoch: the last at the epoch's end, the others at
    # evenly spaced shuffle-window boundaries. Early stopping and the learning
    # rate schedule count rounds, so patience is in rounds, not epochs.
    validations = getattr(args, 'validations_per_epoch', 1)
    settings['validations_per_epoch'] = validations
    patience,min_delta=getattr(args,'early_stopping_patience',0),getattr(args,'min_delta',0.0)
    if patience:settings['early_stopping']=dict(patience=patience,min_delta=min_delta)
    # The rate at every update (LearningRate): warmup, an optional cosine over the planned
    # run, and a plateau decay counted in validation rounds.
    lr_decay,min_lr=getattr(args,'lr_decay',1.0),getattr(args,'min_lr',0.0)
    lr_schedule,lr_horizon=getattr(args,'lr_schedule','constant'),getattr(args,'lr_horizon_updates',0)
    if lr_schedule=='cosine' and not lr_horizon:
        # The planned run: every epoch of the ceiling, or the update limit when that is sooner.
        if not args.epochs and not args.max_updates:
            raise ValueError('A cosine schedule needs --epochs, --max-updates or --lr-horizon-updates')
        lr_horizon=(args.epochs*updates_per_epoch(dataset,args.seed,args.batch_size,settings['buffer_bytes'])
                    if args.epochs else args.max_updates)
        if args.max_updates:lr_horizon=min(lr_horizon,args.max_updates)
    rate_options=dict(warmup=getattr(args,'lr_warmup_updates',0),schedule=lr_schedule,
                      horizon=lr_horizon if lr_schedule=='cosine' else 0,min_lr=min_lr,decay=lr_decay,
                      patience=getattr(args,'lr_patience',2),cooldown=getattr(args,'lr_cooldown',0))
    rates=LearningRate(args.lr,**rate_options)
    if rates.settings():settings['lr_schedule']=rates.settings()
    # Refuse a run whose shuffle window cannot supply the requested batch size (batch_fill).
    minimum_fill=getattr(args,'min_batch_fill',0)
    if minimum_fill:
        fill=batch_fill(dataset,args.seed,args.batch_size,settings['buffer_bytes'])
        if leader:print(json.dumps({'batch_fill':fill,'batch_size':args.batch_size,'buffer_mib':args.buffer_mib}),flush=True)
        if fill<minimum_fill:
            raise ValueError(f'Updates would average {fill*args.batch_size:.0f} rows, not the requested {args.batch_size}: '
                             f'the {args.buffer_mib} MiB shuffle window holds too few rows. Raise --buffer-mib '
                             f'(or lower --batch-size); --min-batch-fill 0 disables this check')
    # Early stopping, LR decay and best.npz follow this validation score (SELECT_ON).
    select_on=getattr(args,'select_on','all')
    if select_on not in SELECT_ON:raise ValueError('Unknown validation selection '+str(select_on))
    if select_on!='all':settings['select_on']=select_on
    # Fine-tuning: weights from this checkpoint, fresh optimizer and schedule.
    init=getattr(args,'init',None)
    validate_only=getattr(args,'validate_only',False)
    if validate_only and not init:raise ValueError('--validate-only scores an --init checkpoint')
    if init:settings['init_sha256']=_digest(Path(init))
    stopper=EarlyStopping(patience,min_delta)
    if (args.epochs<0 or (args.epochs==0 and patience==0)
            or min(args.checkpoint_seconds, args.buffer_mib, args.batch_size, args.threads, args.pair_budget) <= 0
            or not math.isfinite(args.checkpoint_seconds) or args.max_updates < 0 or args.seed < 0
            or not 0 <= settings['prefetch_batches'] <= 2 or settings['reader_processes'] < 1
            or type(validations) is not int or validations < 1 or args.batch_size < ranks.world):
        raise ValueError('Invalid run budget')
    if (not math.isfinite(args.lr) or args.lr <= 0 or args.min_free_ram_mib < 1
            or args.max_private_mib < 1 or not 0 < lr_decay <= 1 or not 0 <= min_lr <= args.lr
            or (lr_decay != 1.0 and not patience)):
        raise ValueError('Invalid optimizer/model/resource configuration')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    # validations_done: this epoch's mid-epoch rounds already run.
    epoch, cursor, validations_done, previous = 0, (0, 0), 0, {}
    if args.resume:
        reference, optimizer, document = NeuralPolicy.load(args.resume)
        previous = document['metadata']
        if previous['event_dataset_sha256'] != dataset.digest or previous['event_settings'] != settings:
            raise ValueError('Resume requires identical data, sampler, optimizer and runtime settings')
        epoch = previous['event_cursor']['epoch']
        cursor = tuple(previous['event_cursor']['position'])
        validations_done = previous['event_cursor']['validations']
        stopper=EarlyStopping(patience,min_delta,previous.get('event_early_stopping'))
        rates=LearningRate(args.lr,**rate_options,state=previous.get('event_lr_state'))
        if leader:
            restore_best_checkpoint(args.resume,best,stopper,previous)
        ranks.barrier()
    elif init:
        reference, _, _ = NeuralPolicy.load(init)
        if reference.config != dataset.config:
            raise ValueError('--init checkpoint has a different model or feature configuration than the dataset')
        optimizer = Adam(reference.parameters, args.lr)
    else:
        reference, optimizer = NeuralPolicy(dataset.config, args.seed), None
        optimizer = Adam(reference.parameters, args.lr)
    if feature_identity(reference) != dataset.key:
        raise ValueError('Event data use a different feature/label implementation')
    learner = EventLearner(reference, optimizer, device=args.device, threads=args.threads, pair_budget=args.pair_budget,
                           precision=settings['precision'], compile_encoder=settings['compile'], ranks=ranks)
    if ranks.world > 1:  # Same seed or checkpoint: every rank must start from identical weights.
        weights = hashlib.sha256()
        for name in sorted(reference.parameters):
            weights.update(name.encode()); weights.update(np.ascontiguousarray(reference.parameters[name].data).tobytes())
        ranks.same(weights.hexdigest())
    # Inline and reader-process preparation use this same specification.
    spec = PreparationSpec(dataset.config, dataset.key, float_dtype=learner.model.float_dtype,
                           device_type=learner.model.device_type)
    if not within_budget(snapshot(output, learner.model), **guard):
        raise RuntimeError('Event model allocation reached its resource boundary; rerun to resume')
    if previous:
        learner.restore_rng(previous)
        learner.restore_scaler(previous)
    source_proofs = [entry['index']['source'] for entry in dataset.entries]
    sources = Path(__file__).parent
    names = ('train_events.py', 'event_resources.py', 'event_training_batches.py', 'event_policy.py',
             'event_schedule.py', 'build_placement.py', 'unit_tasks.py', 'event_store.py', 'event_dataset.py', 'replay_events.py',
             'generals_policy.py', 'torch_policy_v3.py', 'torch_training_inputs.py', 'generals_identity.py',
             'nn_numpy.py', 'training_legality.py', 'training_preparation.py', 'tensor_replay_store.py', 'prepared_io.py',
             'replay_sequences.py', 'replay_actions.py')
    code_hashes = {name: _digest(sources/name) for name in names}
    if previous and previous.get('event_code') != code_hashes:
        raise ValueError('Event training implementation changed; start a separately identified run')
    metadata = {'event_dataset_sha256': dataset.digest, 'event_settings': settings,
                'event_code': code_hashes, 'sources': source_proofs,
                'deployment': dataset.document.get('deployment'), 'event_cursor': {}}
    metadata['event_match_partitions'] = {e['match_key']: e['split'] for e in dataset.entries}
    metadata['event_resource_limits'] = guard
    if not previous and leader:
        metadata['event_cursor'] = {'epoch': 0, 'position': [0, 0], 'validations': 0}
        learner.save(output/'initial.npz', metadata)
        with zipfile.ZipFile(output/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(sources/name, name)
    history = list(previous.get('event_history', []))
    # Training counts since the last validation round. Saved counts are already
    # merged across ranks, so only rank 0 carries them on.
    round_counts = Counter(previous.get('event_round_counts', {}) if leader else {})
    share = {'share': ranks.share} if ranks.share else {}
    counts, start, checkpoint_time = Counter(), time.monotonic(), time.monotonic()
    setup_seconds = start-invocation_start
    stop_requested = False
    stop_file = Path(args.stop_file) if getattr(args, 'stop_file', None) else None

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    original_handler = signal.signal(signal.SIGINT, request_stop)

    def checkpoint():
        """Every rank calls this at the same update; rank 0 writes."""
        metadata['event_cursor'] = {'epoch': epoch, 'position': list(cursor), 'validations': validations_done}
        metadata['event_history'] = history
        metadata['event_round_counts'] = dict(ranks.merge(round_counts))
        metadata['event_early_stopping'] = stopper.state()
        metadata['event_lr_state'] = rates.state()
        if getattr(learner, 'outliers', None):  # Each rank its own file: outliers are reports, not run state.
            name = 'outliers.jsonl' if ranks.world == 1 else f'outliers-rank{ranks.rank}.jsonl'
            with open(output/name, 'a') as stream:
                stream.writelines(json.dumps(record)+'\n' for record in learner.outliers)
            learner.outliers.clear()
        if not leader:
            return
        # latest.npz stays in place until its successor is complete.
        learner.save(pending, metadata)
        if latest.exists():
            os.replace(latest, output/'previous.npz')
        os.replace(pending, latest)

    def stopping():
        return stop_requested or (stop_file and stop_file.exists())

    def train_segment(mark):
        """Train from the cursor until shuffle window `mark` (None: the epoch's end).

        True when the epoch's data ran out; otherwise stopped at the mark, or
        with the reason in status. The batch that reaches the mark is left for
        the stream reopened after validation, as a resume would leave it.
        """
        nonlocal cursor, status, checkpoint_time
        stream = learner.staged(prepared_batches(dataset, 'train', spec=spec, pair_budget=args.pair_budget,
                reader=reader, seed=args.seed, epoch=epoch, batch_size=args.batch_size,
                buffer_bytes=settings['buffer_bytes'], cursor=cursor, **share))
        try:
            for update_inputs, following in stream:
                # following names the batch's own shuffle window.
                if mark is not None and following[0] >= mark:
                    return False
                resources = snapshot(output, learner.model)
                interrupted, short = ranks.any(stopping(), not within_budget(resources, **guard))
                if interrupted:
                    status = 'interrupted'
                    return False
                if not resource_log or time.monotonic()-start-resource_log[-1]['seconds'] >= 10 or short:
                    resource_log.append({'seconds': time.monotonic()-start, **resources})
                if short:
                    status = 'resource_limit'
                    return False
                learner.set_lr(rates.at(learner.step_count))
                try:
                    update = learner.run(update_inputs, train=True)
                except learner.torch.OutOfMemoryError:
                    update = None  # Retried outside the handler, so the traceback's tensors are freed first.
                if update is None:
                    # Under the Windows/HIP cap a fragmented cache can refuse an allocation that fits once the
                    # cached blocks are handed back (16 Nov 2026: 9.0 GB allocated + 2.0 GB cached-but-free, 1.07
                    # GB asked). One flush and retry; a second refusal is a real shortage and raises.
                    learner.optimizer.zero_grad(set_to_none=True)
                    learner.torch.cuda.empty_cache()
                    print('out of memory: flushed the allocator cache and retried the update', file=sys.stderr, flush=True)
                    update = learner.run(update_inputs, train=True)
                counts.update(update); round_counts.update(update)
                cursor = following
                interrupted, due = ranks.any(stop_requested, time.monotonic()-checkpoint_time >= args.checkpoint_seconds)
                if interrupted:
                    status = 'interrupted'
                    return False
                if due:
                    checkpoint(); checkpoint_time = time.monotonic()
                if args.max_updates and learner.step_count >= args.max_updates:
                    status = 'update_limit'
                    return False
            return True
        finally:
            stream.close()

    def validation_round(fraction):
        """Validate, record the round and adjust the schedule; None when interrupted, else improved."""
        nonlocal status, round_counts, resumed_round
        cuda = learner.model.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize()
            # Hand training's cached blocks back before validation's differently shaped batches, and validation's
            # back after it. Windows/HIP never fails an allocation past VRAM: it spills to system RAM (up to 11 GB
            # on the 12 GB Radeon, 16 Nov 2026), so the caching allocator never frees on its own.
            torch.cuda.empty_cache()
        train_seconds = time.monotonic()-round_start
        validation_start = time.monotonic()
        validation = Counter()
        stream = learner.staged(prepared_batches(dataset, 'validation', spec=spec, pair_budget=args.pair_budget,
                reader=reader, seed=args.seed, epoch=0, batch_size=args.batch_size,
                buffer_bytes=settings['buffer_bytes'], **share))
        try:
            for update_inputs, _ in stream:
                resources = snapshot(output, learner.model)
                interrupted, short = ranks.any(stopping(), not within_budget(resources, **guard))
                if interrupted or short:
                    status = 'interrupted' if interrupted else 'resource_limit'
                    resource_log.append({'seconds': time.monotonic()-start, **resources})
                    return None
                validation.update(learner.run(update_inputs, train=False))
        finally:
            stream.close()
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        validation = ranks.merge(validation)
        if not validation['decisions'] and patience:
            raise ValueError('Validation split contains no supervised decisions')
        history.append({'epoch': epoch+1, 'round': len(history)+1, 'epoch_fraction': fraction,
                        'train': summarize(ranks.merge(round_counts)), 'validation': summarize(validation),
                        'seconds': time.monotonic()-round_start, 'train_seconds': train_seconds,
                        'validation_seconds': time.monotonic()-validation_start,
                        'timing_scope': 'remaining_round_after_resume' if resumed_round else 'full_round',
                        'lr': learner.lr})
        score = selection_score(history[-1]['validation'], select_on)
        if select_on != 'all':
            history[-1]['selection_score'] = score
        improved = stopper.observe(len(history), score) if score is not None else False
        if not stopper.stopped and rates.observe(improved):
            history[-1]['next_lr'] = rates.at(learner.step_count)
        if rates.settings():history[-1]['lr_state'] = rates.state()
        round_counts, resumed_round = Counter(), False
        return improved

    def publish(improved):
        """Checkpoint right after a round; that cursor identifies the round for best-copy recovery."""
        nonlocal checkpoint_time, round_start
        history[-1]['cursor'] = {'epoch': epoch, 'position': list(cursor), 'validations': validations_done}
        saving = time.monotonic()
        checkpoint(); checkpoint_time = time.monotonic()
        if improved and leader:copy_checkpoint(latest,best)
        history[-1]['checkpoint_seconds'] = time.monotonic()-saving
        if leader:
            print(json.dumps(history[-1]), flush=True)
        round_start = time.monotonic()

    status = 'completed'
    resource_log = []
    round_start, resumed_round = time.monotonic(), bool(round_counts)
    reader = (PreparedReader(dataset, spec, args.pair_budget, settings['prefetch_batches'], settings['reader_processes'],
                             ranks.share) if settings['prefetch_batches'] else None)
    try:
        if validate_only:  # Score --init on this dataset's validation split; no updates.
            if validation_round(0.0) is not None:
                status = 'validated'
        while not validate_only and (not args.epochs or epoch < args.epochs):
            if stopper.stopped:
                status='early_stopped'
                break
            if args.max_updates and learner.step_count >= args.max_updates:
                status = 'update_limit'
                break
            marks = validation_marks(dataset, args.seed, epoch, settings['buffer_bytes'], validations)
            exhausted = False
            # Mid-epoch rounds: train to each mark, then validate there.
            while status == 'completed' and not exhausted and not stopper.stopped:
                mark = marks[validations_done] if validations_done < len(marks) else None
                exhausted = train_segment(mark)
                if status == 'completed' and not exhausted:
                    improved = validation_round((validations_done+1)/validations)
                    if improved is not None:
                        validations_done += 1
                        publish(improved)
            if status != 'completed' or not exhausted:
                break
            improved = validation_round(1.0)
            if improved is None:
                break
            epoch += 1; cursor = (0, 0); validations_done = 0
            publish(improved)
    finally:
        if reader is not None:
            reader.close()
        signal.signal(signal.SIGINT, original_handler)
    if status=='completed' and stopper.stopped:status='early_stopped'
    checkpoint()
    counts = ranks.merge(counts)
    if not leader:
        return None
    report = {'status': status, 'seconds': time.monotonic()-invocation_start, 'setup_seconds': setup_seconds,
              'optimizer_updates': learner.step_count,
              'this_invocation': summarize(counts), 'history': history, 'settings': settings,
              'cursor': metadata['event_cursor'], 'dataset_sha256': dataset.digest,
              'training_matches': sum(e['split'] == 'train' for e in dataset.entries),
              'validation_matches': sum(e['split'] == 'validation' for e in dataset.entries),
              'checkpoint': str(latest), 'checkpoint_sha256': _digest(latest),
              'early_stopping': stopper.state(),
              'best_checkpoint': str(best) if best.exists() else None,
              'best_checkpoint_sha256': _digest(best) if best.exists() else None,
              'staging': learner.model.stager.stats, 'grad_scale': learner.scaler.get_scale() if learner.fp16 else None, 'resources': resource_log,
              'final_resources': snapshot(output, learner.model),
              'torch_version': torch.__version__, 'hip_version': torch.version.hip,
              'gpu': torch.cuda.get_device_name(learner.model.device) if learner.model.device.type == 'cuda' else None}
    report['resource_limits'] = guard
    _write_json(output/'report.json', report)
    print(json.dumps(report), flush=True)
    return report


def _rank_main(function, args, rank, world, address, ranks_per_gpu):
    from engine_runtime import disable_power_throttling
    disable_power_throttling()  # Spawned ranks are windowless; the opt-out is not inherited.
    backend = None
    if args.device == 'cuda':
        import torch
        # 'cuda' then means this rank's device in all model and staging code.
        torch.cuda.set_device(rank//ranks_per_gpu)
        if ranks_per_gpu > 1:
            backend = 'gloo'  # NCCL refuses two ranks on one device.
    ranks = DistributedRanks(rank, world, address, args.device, backend)
    try:
        return function(args, ranks)
    finally:
        ranks.close()


def spawn_ranks(function, args, world, ranks_per_gpu=1):
    """Run function(args, ranks) in `world` data-parallel processes on this host.

    Any rank failing stops the rest, so none waits forever in a collective.
    function must be importable by name (spawn pickles a reference).
    """
    import multiprocessing
    import socket
    if world < 2 or ranks_per_gpu < 1 or world % ranks_per_gpu:
        raise ValueError('Distributed runs need two or more ranks, a whole number per GPU')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    # libuv is missing from Windows builds; the plain TCP store works on both.
    address = f'tcp://127.0.0.1:{port}?use_libuv=0'
    context = multiprocessing.get_context('spawn')
    processes = [context.Process(target=_rank_main, args=(function, args, rank, world, address, ranks_per_gpu),
                                 name=f'event-rank-{rank}') for rank in range(world)]
    # Ctrl+C reaches every rank, which checkpoint and stop together; the launcher just waits.
    original = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            failed = [p for p in processes if p.exitcode not in (None, 0)]
            if failed:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join()
                raise RuntimeError(f'{failed[0].name} exited with code {failed[0].exitcode}; all ranks stopped')
            time.sleep(.2)
        codes = [process.exitcode for process in processes]
        if any(codes):
            raise RuntimeError(f'Ranks exited with codes {codes}')
    finally:
        signal.signal(signal.SIGINT, original)


def launch(args, world):
    """Train with `world` ranks, one GPU each; returns rank 0's report."""
    spawn_ranks(train, args, world)
    report = Path(args.output)/'report.json'
    return json.loads(report.read_text()) if report.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--dataset', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--resume')
    run.add_argument('--init', help='Start from this checkpoint\'s weights (fine-tuning): fresh optimizer, --lr and schedule')
    run.add_argument('--select-on', choices=tuple(SELECT_ON), default='all',
                     help='Validation score for early stopping, LR decay and best.npz: all decisions, players rated 1400+, '
                          'or vanilla-general players only')
    run.add_argument('--validate-only', action='store_true',
                     help='Score the --init checkpoint on the validation split (report.json history[0]) without training')
    run.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    run.add_argument('--epochs', type=int, default=1,help='Epoch ceiling; 0 requires early stopping and runs until plateau')
    run.add_argument('--early-stopping-patience', type=int, default=0,
                     help='Stop after this many epochs without lower validation NLL; 0 disables')
    run.add_argument('--min-delta', type=float, default=0.0)
    run.add_argument('--validations-per-epoch', type=int, default=1,
                     help='Validation rounds per epoch, evenly spaced; patience and LR decay count rounds')
    run.add_argument('--max-updates', type=int, default=0)
    run.add_argument('--seed', type=int, default=123)
    run.add_argument('--lr', type=float, default=3e-4)
    run.add_argument('--lr-decay', type=float, default=1.0,
                     help='Plateau factor: multiply the rate by this after --lr-patience validation rounds in a row '
                          'without improvement (1 disables)')
    run.add_argument('--lr-patience', type=int, default=2,
                     help='Consecutive non-improving validation rounds before a plateau decay; 1 reacts to a single round')
    run.add_argument('--lr-cooldown', type=int, default=0, help='Validation rounds not counted after a plateau decay')
    run.add_argument('--lr-warmup-updates', type=int, default=0, help='Linear ramp to --lr over this many updates')
    run.add_argument('--lr-schedule', choices=LearningRate.SCHEDULES, default='constant',
                     help='cosine: from --lr to --min-lr over the planned run (--epochs, or --lr-horizon-updates)')
    run.add_argument('--lr-horizon-updates', type=int, default=0, help='Cosine length in updates; 0 takes it from --epochs')
    run.add_argument('--min-lr', type=float, default=0.0, help='Floor of the cosine and of plateau decays')
    run.add_argument('--prefetch-batches', type=int, default=1,
                     help='Prepared updates queued by the reader process (0 prepares inline; maximum 2)')
    run.add_argument('--precision', choices=PRECISIONS, default='fp32',
                     help='fp32 strict; tf32 (NVIDIA) fast FP32 matmuls; bf16/fp16 autocast trunk, FP32 master weights')
    run.add_argument('--compile', action='store_true',
                     help='torch.compile the encoder (NVIDIA CUDA with Triton); speed only')
    run.add_argument('--reader-processes', type=int, default=1,
                     help='Parallel shuffle-window readers; batches and order are unchanged')
    run.add_argument('--min-free-disk-gib', type=int, default=80)
    run.add_argument('--max-gpu-mib', type=int, default=9216, help='Stop and checkpoint above this allocation')
    run.add_argument('--threads', type=int, default=1)
    run.add_argument('--batch-size', type=int, default=256,
                     help='Global batch: with --world-size N each rank trains on every N-th row')
    run.add_argument('--world-size', type=int, default=1,
                     help='Data-parallel ranks on this host, one GPU each (gloo on CPU for testing)')
    run.add_argument('--pair-budget', type=int, default=8_000_000)
    run.add_argument('--buffer-mib', type=int, default=64,
                     help='Shuffle window; batches are cut from one window, so it must hold many batches')
    run.add_argument('--min-batch-fill', type=float, default=0.9,
                     help='Refuse to start when updates would average less than this share of --batch-size '
                          '(the shuffle window is too small); 0 disables')
    run.add_argument('--checkpoint-seconds', type=float, default=600)
    run.add_argument('--stop-file', help='Checkpoint and exit at an update boundary when this file exists')
    run.add_argument('--min-free-ram-mib', type=int, default=4096)
    run.add_argument('--max-private-mib', type=int, default=8192)
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error('--world-size must be positive')
    if args.world_size == 1:
        train(args)
    else:
        if args.device == 'cuda':
            import torch
            if torch.cuda.device_count() < args.world_size:
                parser.error(f'--world-size {args.world_size} needs that many GPUs; found {torch.cuda.device_count()}')
        launch(args, args.world_size)


if __name__ == '__main__':
    main()
