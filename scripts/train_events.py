"""Train the memory-free event policy on prepared event shards."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import signal
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

# Game-time buckets (minutes) for validation by phase: late-game coverage is thin.
PHASES = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, None))


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


def phase(row):
    minutes = row['frame']/row.get('source_ticks_per_second', 60)/60
    low, high = next(bounds for bounds in PHASES if bounds[1] is None or minutes < bounds[1])
    return f'{low:02d}-{high:02d}' if high is not None else f'{low:02d}+'


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
                 precision='fp32', compile_encoder=False):
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
        self.torch, self.reference = torch, reference
        self.model = TorchPolicyV3(reference, device=device, dtype=torch.float32)
        if compile_encoder:
            # Fuses the entity/map encoder, where most launched ops are (+12-25%
            # on an RTX 5070 Ti); heads stay eager. Dynamic shapes: every
            # microbatch has its own batch and entity counts.
            import importlib.util
            if device != 'cuda' or torch.version.hip is not None or importlib.util.find_spec('triton') is None:
                raise ValueError('--compile needs an NVIDIA CUDA device with Triton')
            stride_order_with_shape_env()
            self.model.encode_inputs = torch.compile(self.model.encode_inputs, dynamic=True)
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
        decisions, delays, losses_by_part = [], [], []
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            for part in prepared['microbatches']:
                rows = part['rows']
                # FP16 covers the trunk, where nearly all the work is; the small
                # argument heads and losses stay FP32.
                with torch.autocast('cuda', dtype=self.autocast or torch.float16, enabled=self.autocast is not None):
                    trunk = model.events(prepared=part)
                hidden, entities, spatial = (value.float() for value in trunk)
                del trunk
                output = model.teacher_heads(None, [row['target'] for row in rows], hidden,
                                             entities, spatial, loss_only=True, plan=part['plan'])
                losses = event_losses(model, rows, output)
                if train:
                    self.scaler.scale(losses.sum()/prepared['rows']).backward()
                decisions.append(output['predictions'][0][2].detach())
                delays.append(next(item[2] for item in output['predictions'] if item[0] == 'delay').detach())
                losses_by_part.append(losses.detach())
                # Release the preceding microbatch's graph before allocating
                # the next one's tensors.
                del hidden, entities, spatial, output, losses
            norm = None
            if train:
                regularization = model.regularization_loss()
                if regularization.requires_grad:
                    self.scaler.scale(regularization).backward()
                self.scaler.unscale_(self.optimizer)
                # Under loss scaling an overflow is expected occasionally: the
                # scaler skips that step and lowers its scale instead.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=not self.fp16,
                                                      foreach=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step_count += 1
                totals['updates'] = 1
        guesses = torch.cat([torch.stack([d, e]) for d, e in zip(decisions, delays)], dim=1).cpu().numpy()
        values = torch.cat(losses_by_part+([norm.reshape(1).to(losses_by_part[0].dtype)] if norm is not None else []))
        values = values.cpu().numpy()
        start = 0
        for part, row_losses in zip(prepared['microbatches'], np.split(values, np.cumsum([len(p['rows']) for p in
                                                                                       prepared['microbatches']]))):
            rows = part['rows']
            self._count(totals, rows, guesses[0, start:start+len(rows)], guesses[1, start:start+len(rows)], row_losses)
            start += len(rows)
            totals['microbatches'] += 1
            totals['entity_pair_work'] += len(rows)*part['entity_count']**2
        if norm is not None:
            norm = float(values[-1])
            if math.isfinite(norm):
                totals['gradient_norm'] = norm
            else:
                totals['skipped_updates'] = 1
        return totals

    def _count(self, totals, rows, predicted, delay_predictions, row_losses):
        """Per-row decision/delay statistics of one microbatch, vectorised; same sums and order."""
        kinds = self.model.reference.kinds
        targets = [row['target'] for row in rows]
        labels = np.fromiter((kinds.index(t['kind']) for t in targets), dtype=np.int64, count=len(rows))
        hits = predicted == labels
        totals['loss_sum'] += float(row_losses.sum())
        totals['decisions'] += len(rows)
        totals['correct'] += int(hits.sum())
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
        steps = {}
        for name, parameter in self.model.weights.items():
            self.reference.parameters[name].data[:] = parameter.detach().cpu().numpy()
            state = self.optimizer.state.get(parameter)
            steps[name] = int(state['step'].item()) if state else 0
            if state:
                self.mirror.m[name][:] = state['exp_avg'].detach().cpu().numpy()
                self.mirror.v[name][:] = state['exp_avg_sq'].detach().cpu().numpy()
        self.mirror.step_count = self.step_count
        rng = {'torch': self.torch.get_rng_state().tolist(),
               'cuda': [state.tolist() for state in self.torch.cuda.get_rng_state_all()] if self.torch.cuda.is_available() else [],
               'python': random.getstate(),
               'numpy': [item.tolist() if isinstance(item, np.ndarray) else item for item in np.random.get_state()]}
        if self.fp16:
            metadata = {**metadata, 'event_grad_scaler': self.scaler.state_dict()}
        self.reference.save(path, self.mirror, {**metadata, 'event_rng': rng, 'event_optimizer_steps': steps})

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


def summarize(counts):
    phases = sorted({key.split(':')[1] for key in counts if key.startswith('minute:')})
    by_minute = {name: {'decisions': counts['minute:'+name+':decisions'],
                        'mean_nll': counts['minute:'+name+':loss_sum']/max(1, counts['minute:'+name+':decisions']),
                        'decision_accuracy': counts['minute:'+name+':correct']/max(1, counts['minute:'+name+':decisions'])}
                 for name in phases}
    return {**{k: v for k, v in counts.items() if not k.startswith('minute:')}, 'by_minute': by_minute,
            'mean_nll': counts['loss_sum']/max(1, counts['decisions']),
            'decision_accuracy': counts['correct']/max(1, counts['decisions']),
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


def train(args):
    invocation_start = time.monotonic()
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
    best = output/'best.npz'
    if (latest.exists() or best.exists()) and not args.resume:
        raise ValueError('Output already has a checkpoint; use --resume explicitly')
    settings = {'seed': args.seed, 'batch_size': args.batch_size, 'buffer_bytes': args.buffer_mib*1024*1024,
                'pair_budget': args.pair_budget, 'device': args.device, 'threads': args.threads, 'lr': args.lr,
                'prefetch_batches': getattr(args, 'prefetch_batches', 1),
                'reader_processes': getattr(args, 'reader_processes', 1),
                'precision': getattr(args, 'precision', 'fp32'), 'compile': getattr(args, 'compile', False)}
    # Validation rounds per epoch: the last at the epoch's end, the others at
    # evenly spaced shuffle-window boundaries. Early stopping and the learning
    # rate schedule count rounds, so patience is in rounds, not epochs.
    validations = getattr(args, 'validations_per_epoch', 1)
    settings['validations_per_epoch'] = validations
    patience,min_delta=getattr(args,'early_stopping_patience',0),getattr(args,'min_delta',0.0)
    if patience:settings['early_stopping']=dict(patience=patience,min_delta=min_delta)
    # Reduce-on-plateau: every validation round without an improvement
    # multiplies the learning rate by lr_decay (1 disables), down to min_lr.
    lr_decay,min_lr=getattr(args,'lr_decay',1.0),getattr(args,'min_lr',0.0)
    if lr_decay!=1.0:settings['lr_schedule']=dict(decay=lr_decay,min_lr=min_lr)
    stopper=EarlyStopping(patience,min_delta)
    if (args.epochs<0 or (args.epochs==0 and patience==0)
            or min(args.checkpoint_seconds, args.buffer_mib, args.batch_size, args.threads, args.pair_budget) <= 0
            or not math.isfinite(args.checkpoint_seconds) or args.max_updates < 0 or args.seed < 0
            or not 0 <= settings['prefetch_batches'] <= 2 or settings['reader_processes'] < 1
            or type(validations) is not int or validations < 1):
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
        restore_best_checkpoint(args.resume,best,stopper,previous)
    else:
        reference, optimizer = NeuralPolicy(dataset.config, args.seed), None
        optimizer = Adam(reference.parameters, args.lr)
    if feature_identity(reference) != dataset.key:
        raise ValueError('Event data use a different feature/label implementation')
    learner = EventLearner(reference, optimizer, device=args.device, threads=args.threads, pair_budget=args.pair_budget,
                           precision=settings['precision'], compile_encoder=settings['compile'])
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
    if not previous:
        metadata['event_cursor'] = {'epoch': 0, 'position': [0, 0], 'validations': 0}
        learner.save(output/'initial.npz', metadata)
        with zipfile.ZipFile(output/'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(sources/name, name)
    history = list(previous.get('event_history', []))
    # Training counts since the last validation round.
    round_counts = Counter(previous.get('event_round_counts', {}))
    counts, start, checkpoint_time = Counter(), time.monotonic(), time.monotonic()
    setup_seconds = start-invocation_start
    stop_requested = False
    stop_file = Path(args.stop_file) if getattr(args, 'stop_file', None) else None

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    original_handler = signal.signal(signal.SIGINT, request_stop)

    def checkpoint():
        metadata['event_cursor'] = {'epoch': epoch, 'position': list(cursor), 'validations': validations_done}
        metadata['event_history'] = history
        metadata['event_round_counts'] = dict(round_counts)
        metadata['event_early_stopping'] = stopper.state()
        if latest.exists():
            os.replace(latest, output/'previous.npz')
        learner.save(latest, metadata)

    def stopping():
        return stop_requested or (stop_file and stop_file.exists())

    def train_segment(mark):
        """Train from the cursor until shuffle window `mark` (None: the epoch's end).

        True when the epoch's data ran out; otherwise stopped at the mark, or
        with the reason in status. The batch that reaches the mark is left for
        the stream reopened after validation, as a resume would leave it.
        """
        nonlocal cursor, status, checkpoint_time
        stream = prepared_batches(dataset, 'train', spec=spec, pair_budget=args.pair_budget, reader=reader,
                seed=args.seed, epoch=epoch, batch_size=args.batch_size,
                buffer_bytes=settings['buffer_bytes'], cursor=cursor)
        try:
            for update_inputs, following in stream:
                # following names the batch's own shuffle window.
                if mark is not None and following[0] >= mark:
                    return False
                if stopping():
                    status = 'interrupted'
                    return False
                resources = snapshot(output, learner.model)
                if not resource_log or time.monotonic()-start-resource_log[-1]['seconds'] >= 10 or not within_budget(resources, **guard):
                    resource_log.append({'seconds': time.monotonic()-start, **resources})
                if not within_budget(resources, **guard):
                    status = 'resource_limit'
                    return False
                update = learner.run(update_inputs, train=True)
                counts.update(update); round_counts.update(update)
                cursor = following
                if stop_requested:
                    status = 'interrupted'
                    return False
                if time.monotonic()-checkpoint_time >= args.checkpoint_seconds:
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
        if learner.model.device.type == 'cuda':
            torch.cuda.synchronize()
        train_seconds = time.monotonic()-round_start
        validation_start = time.monotonic()
        validation = Counter()
        stream = prepared_batches(dataset, 'validation', spec=spec, pair_budget=args.pair_budget,
                reader=reader, seed=args.seed, epoch=0, batch_size=args.batch_size,
                buffer_bytes=settings['buffer_bytes'])
        try:
            for update_inputs, _ in stream:
                resources = snapshot(output, learner.model)
                if not within_budget(resources, **guard) or stopping():
                    status = 'interrupted' if stopping() else 'resource_limit'
                    resource_log.append({'seconds': time.monotonic()-start, **resources})
                    return None
                validation.update(learner.run(update_inputs, train=False))
        finally:
            stream.close()
        if not validation['decisions'] and patience:
            raise ValueError('Validation split contains no supervised decisions')
        history.append({'epoch': epoch+1, 'round': len(history)+1, 'epoch_fraction': fraction,
                        'train': summarize(round_counts), 'validation': summarize(validation),
                        'seconds': time.monotonic()-round_start, 'train_seconds': train_seconds,
                        'validation_seconds': time.monotonic()-validation_start,
                        'timing_scope': 'remaining_round_after_resume' if resumed_round else 'full_round',
                        'lr': learner.lr})
        improved = (stopper.observe(len(history), history[-1]['validation']['mean_nll'])
                    if validation['decisions'] else False)
        if not improved and lr_decay!=1.0 and not stopper.stopped:
            learner.set_lr(max(min_lr,learner.lr*lr_decay))
            history[-1]['next_lr'] = learner.lr
        round_counts, resumed_round = Counter(), False
        return improved

    def publish(improved):
        """Checkpoint right after a round; that cursor identifies the round for best-copy recovery."""
        nonlocal checkpoint_time, round_start
        history[-1]['cursor'] = {'epoch': epoch, 'position': list(cursor), 'validations': validations_done}
        saving = time.monotonic()
        checkpoint(); checkpoint_time = time.monotonic()
        if improved:copy_checkpoint(latest,best)
        history[-1]['checkpoint_seconds'] = time.monotonic()-saving
        print(json.dumps(history[-1]), flush=True)
        round_start = time.monotonic()

    status = 'completed'
    resource_log = []
    round_start, resumed_round = time.monotonic(), bool(round_counts)
    reader = (PreparedReader(dataset, spec, args.pair_budget, settings['prefetch_batches'], settings['reader_processes'])
              if settings['prefetch_batches'] else None)
    try:
        while not args.epochs or epoch < args.epochs:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--dataset', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--resume')
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
                     help='Multiply the learning rate by this after each epoch without validation improvement (1 disables)')
    run.add_argument('--min-lr', type=float, default=0.0)
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
    run.add_argument('--batch-size', type=int, default=256)
    run.add_argument('--pair-budget', type=int, default=8_000_000)
    run.add_argument('--buffer-mib', type=int, default=64)
    run.add_argument('--checkpoint-seconds', type=float, default=600)
    run.add_argument('--stop-file', help='Checkpoint and exit at an update boundary when this file exists')
    run.add_argument('--min-free-ram-mib', type=int, default=4096)
    run.add_argument('--max-private-mib', type=int, default=8192)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
