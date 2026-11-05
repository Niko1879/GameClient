"""Portable immutable event columns with source proofs and bounded decoding."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import zipfile

import numpy as np

import generals_policy
from generals_policy import ARCHITECTURE, NeuralPolicy
from prepared_io import _array_bytes, _common, _digest, _json_bytes, _write_json
from replay_events import encoded_events
from tensor_replay_store import Column, TensorRows, decode_columns, encode_columns, project
from training_legality import build_legality

SCHEMA = 1
# Stored inputs, labels and legality depend on these sources and on the policy
# configuration, never on network widths or optimizer settings.
FEATURE_SOURCES = ('replay_events.py', 'event_schedule.py', 'replay_actions.py', 'replay_sequences.py',
                   'generals_identity.py', 'tensor_replay_store.py', 'training_legality.py', 'event_store.py',
                   'build_placement.py', 'unit_tasks.py')
FEATURE_CODE = (generals_policy.entity_key, generals_policy.initial_home, generals_policy.Encoder,
                generals_policy.selection_limit, generals_policy.command_target,
                NeuralPolicy.actor_mask, NeuralPolicy.dock_matrix, NeuralPolicy.resume_matrix,
                NeuralPolicy.enter_matrix, NeuralPolicy.pair_matrix, NeuralPolicy.slot_mask, NeuralPolicy.force_targets,
                NeuralPolicy._commands, NeuralPolicy.command_mask,
                NeuralPolicy.science_mask, NeuralPolicy.command_targets,
                NeuralPolicy.target_mask, NeuralPolicy.selection_mask, NeuralPolicy.verb_mask, NeuralPolicy.target)
FEATURE_CONSTANTS = ('OBSERVATION_VERSION', 'KINDS', 'TARGET_KINDS', 'BUILDER_KINDS', 'GROUP_KINDS', 'SPATIAL_KINDS',
                     'COMMAND_KINDS', 'GUARD_KINDS', 'QUEUE_SLOTS', 'PAIR_FIELDS', 'ENTITY_FEATURES', 'EXTRA_ENTITY_FEATURES', 'ENTITY_UPGRADE_SLOTS',
                     'ENTITY_COMMAND_SLOTS', 'SUPPLY_BOX_SCALE',
                     'SCALAR_FEATURES', 'MAP_CHANNELS')
MODEL_ONLY = {'hidden', 'entity_hidden', 'embedding', 'attention_heads', 'map_hidden', 'scalar_hidden',
              'variant_l2', 'argument_loss_weights', 'delay_loss_weight'}


_FEATURE_CODE = []


def feature_identity(policy):
    # Code is hashed as loaded; source_guard stops the service if its files change.
    if not _FEATURE_CODE:
        _FEATURE_CODE.extend(inspect.getsource(obj) for obj in FEATURE_CODE)
    identity = {'schema': SCHEMA,
                'config': {k: v for k, v in policy.config.items() if k not in MODEL_ONLY},
                'feature_code': _FEATURE_CODE,
                'constants': {name: getattr(generals_policy, name) for name in FEATURE_CONSTANTS},
                'dependencies': {name: _digest(Path(__file__).with_name(name)) for name in FEATURE_SOURCES}}
    return hashlib.sha256(_json_bytes(identity)).hexdigest()


# Chunks are the training shuffle unit; this keeps them near ~1 MB stored.
CHUNK_ARRAY_BYTES = 5*512*1024


def write_shard(destination, games, policy, source):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    key, counts = feature_identity(policy), Counter()
    float_dtype = 'float32'
    index = {'schema': SCHEMA, 'kind': 'generals_event_shard', 'legality_key': key, 'float_dtype': float_dtype,
             'episodes': [], 'shared': {}, 'source': source}
    temporary = destination.with_name(destination.name+'.part')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for number, game in enumerate(games):
                common, buffer, size, chunks, start = None, [], 0, [], 0
                episode_counts = Counter()

                def flush():
                    nonlocal buffer, size, start
                    if not buffer:
                        return
                    for row in buffer:
                        row['encoded']['_training_legality'] = build_legality(policy, row['encoded'], row['target'], key)
                    packed = project(buffer, common, policy, key, float_dtype=float_dtype)
                    for row, meta in zip(buffer, packed.metadata):
                        meta['row'].update({name: row[name] for name in ('frame', 'player', 'source_ticks_per_second')})
                        meta['encoded']['event_rate'] = row['source_ticks_per_second']
                    raw = encode_columns(packed.columns, packed.metadata, float_dtype=float_dtype)
                    member = f'chunks/{number:04d}/{len(chunks):06d}.bin'
                    archive.writestr(member, raw)
                    chunks.append({'member': member, 'rows': len(buffer), 'bytes': len(raw),
                                   'sha256': hashlib.sha256(raw).hexdigest(), 'start': start})
                    start += len(buffer)
                    buffer, size = [], 0

                for row in encoded_events(policy, game, episode_counts):
                    if common is None:
                        common = _common(row)
                    estimate = _array_bytes(row['encoded'])
                    if buffer and (len(buffer) >= 128 or size+estimate > CHUNK_ARRAY_BYTES):
                        flush()
                    buffer.append(row); size += estimate
                flush()
                cutoff = episode_counts.pop('production_scope_cutoff_frame', None)
                counts.update(episode_counts)
                if common is None:
                    continue
                raw = encode_columns({name: Column.build([value], float_dtype=float_dtype) for name, value in common.items()
                                      if isinstance(value, np.ndarray)}, [{}], float_dtype=float_dtype)
                member = f'shared/{number:04d}.bin'
                archive.writestr(member, raw)
                index['shared'][str(number)] = {'member': member, 'bytes': len(raw),
                                              'sha256': hashlib.sha256(raw).hexdigest()}
                episode = {'player': game['player'], 'length': start,
                           'shared': str(number), 'chunks': chunks}
                if cutoff is not None:
                    episode['censored'] = {'reason': 'production_scope', 'frame': cutoff}
                index['episodes'].append(episode)
            index['coverage'] = dict(counts)
            archive.writestr('index.json', _json_bytes(index))
        # Verify publication before exposing the immutable path.
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('Event shard CRC verification failed')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'archive_sha256': _digest(destination), 'bytes': destination.stat().st_size,
            'index_sha256': hashlib.sha256(_json_bytes(index)).hexdigest(), 'index': index}


def _preparation_guard(output, minimum_ram, stopped=None):
    from event_resources import snapshot, within_budget
    if stopped is not None and stopped.is_set():
        raise RuntimeError('Event preparation cancelled after another worker stopped')
    if not within_budget(snapshot(output), minimum_ram=minimum_ram):
        raise RuntimeError('Event preparation stopped at its RAM/disk resource boundary; rerun to resume')


class _PreparationDecoder:
    """Worker-local checks during both metadata auditing and dense replay reads."""
    def __init__(self, decoder, output, minimum_ram, stopped):
        self.decoder, self.output = decoder, output
        self.minimum_ram, self.stopped, self.count = minimum_ram, stopped, 0

    def decode(self, *args, **kwargs):
        if self.count % 128 == 0:
            _preparation_guard(self.output, self.minimum_ram, self.stopped)
        self.count += 1
        return self.decoder.decode(*args, **kwargs)

    def raw_decode(self, *args, **kwargs):
        if self.count % 128 == 0:
            _preparation_guard(self.output, self.minimum_ram, self.stopped)
        self.count += 1
        return self.decoder.raw_decode(*args, **kwargs)


def _initialize_preparation(config, output, minimum_ram, stopped):
    global _PREPARATION_POLICY, _PREPARATION_OUTPUT, _PREPARATION_MINIMUM_RAM, _PREPARATION_STOPPED
    import replay_training
    _PREPARATION_POLICY = NeuralPolicy(config, parameters=False)
    _PREPARATION_OUTPUT, _PREPARATION_MINIMUM_RAM, _PREPARATION_STOPPED = output, minimum_ram, stopped
    # Leave source-bound audit receipts and decoder semantics intact. The
    # adapter exists only in these dedicated worker processes.
    for name in ('_decoder', '_finite_decoder'):
        setattr(replay_training, name, _PreparationDecoder(getattr(replay_training, name), output, minimum_ram, stopped))


def _encode_preparation(job, policy, guard):
    from replay_loading import load_replays
    guard()
    path, destination = Path(job['path']), Path(job['destination'])
    games, proof = job['games'], job['proof']
    if games is None:
        games, _, sources = load_replays([path])
        proof = sources[0]
    elif _digest(path) != proof['sha256']:
        raise ValueError('Training export changed after vocabulary audit')
    original = proof.get('original_replay')
    if original and _digest(original) != proof['replay_sha256']:
        raise ValueError('Original replay changed after vocabulary audit')
    proof = {**proof, 'canonical_export_bytes': path.stat().st_size}
    if proof.get('match_key') != job['match_key'] or proof['sha256'] != job['sha256']:
        raise ValueError('Source proof does not match dataset match ID or export')

    def guarded_rows(rows):
        for index, row in enumerate(rows):
            if index % 128 == 0:
                guard()
            yield row

    guarded_games = [{**game, 'rows': guarded_rows(game['rows'])} for game in games]
    prepared = write_shard(destination, guarded_games, policy, proof)
    if _digest(path) != proof['sha256']:
        raise ValueError('Replay export changed while encoding events')
    _write_json(destination.with_suffix('.json'), prepared)
    guard()
    return prepared


def _preparation_job(job):
    guard = lambda: _preparation_guard(_PREPARATION_OUTPUT, _PREPARATION_MINIMUM_RAM, _PREPARATION_STOPPED)
    return _encode_preparation(job, _PREPARATION_POLICY, guard), os.getpid()


def _check_preparation(destination, prepared, key, match_key, source_digest):
    if _digest(destination) != prepared['archive_sha256']:
        raise ValueError('Existing event shard changed')
    index = prepared['index']
    if (index['legality_key'] != key or index['source'].get('match_key') != match_key
            or index['source'].get('sha256') != source_digest
            or hashlib.sha256(_json_bytes(index)).hexdigest() != prepared['index_sha256']):
        raise ValueError('Prepared features do not match the event contract or source')
    with zipfile.ZipFile(destination) as archive:
        if hashlib.sha256(archive.read('index.json')).hexdigest() != prepared['index_sha256']:
            raise ValueError('Event shard index proof mismatch')


def _prepare_pool(jobs, config, output, minimum_ram, workers, publish):
    context = multiprocessing.get_context('spawn')
    stopped = context.Event()
    # Spawn imports NumPy before the initializer. Limit numerical libraries
    # through the inherited environment, restoring the coordinator afterward.
    names = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    pool = None
    try:
        for name in names:
            os.environ[name] = '1'
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize_preparation, initargs=(config, str(output), minimum_ram, stopped))
        queue = iter(sorted(jobs, key=lambda job: Path(job['path']).stat().st_size, reverse=True))
        pending = {}

        def submit():
            job = next(queue, None)
            if job is not None:
                pending[pool.submit(_preparation_job, job)] = job

        for _ in range(workers):
            submit()
        while pending:
            _preparation_guard(output, minimum_ram)
            done, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                prepared, pid = future.result()
                publish(job, prepared, pid)
                _preparation_guard(output, minimum_ram)
                submit()
    except BaseException:
        stopped.set()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
