"""Bounded prefetch that preserves the event sampler's order and cursors.

`PrefetchedBatches` decodes rows on a thread. `PreparedReader` runs separate
processes (no PyTorch) that decode shards and build the CPU arrays of each
update's lane parts (sequence training, notes/plan.md §2): microbatches, padded
inputs and teacher plans. The learner process then only stages and runs them,
so CPU preparation overlaps GPU work without contending for the learner's
interpreter lock. Readers change speed only, never content.
"""
from __future__ import annotations

import multiprocessing
from multiprocessing import shared_memory
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Event, Thread
import traceback

import numpy as np

from training_preparation import PreparationSpec, prepare_update


class PrefetchedBatches:
    """At most `depth` queued/in-flight batches, plus the consumer's batch.

    Reserve a slot before decoding, so a full queue cannot hide another large
    decoded window. The producer alone touches the dataset iterator. Explicit
    close stops it at epoch boundaries or when unwinding an interruption.
    """
    def __init__(self, iterator, depth=1, *, profiler=None):
        if depth < 1:
            raise ValueError('Positive prefetch depth required')
        self.iterator, self.queue = iter(iterator), Queue(depth)
        self.profiler = profiler
        self.slots, self.stop = BoundedSemaphore(depth), Event()
        self.thread = None

    def _produce(self):
        try:
            while not self.stop.is_set():
                if not self.slots.acquire(timeout=.05):
                    continue
                if self.stop.is_set():
                    break
                try:
                    item = (next(self.iterator) if self.profiler is None else
                            self.profiler.runcall(next, self.iterator))
                except StopIteration:
                    self.queue.put_nowait(('end', None))
                    return
                except BaseException as error:
                    self.queue.put_nowait(('error', error))
                    return
                self.queue.put_nowait(('item', item))
        finally:
            close = getattr(self.iterator, 'close', None)
            if close:
                close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop.is_set():
            raise StopIteration
        if self.thread is None:
            self.thread = Thread(target=self._produce, name='event-batch-reader', daemon=True)
            self.thread.start()
        kind, value = self.queue.get()
        self.slots.release()
        if kind == 'end':
            self.close()
            raise StopIteration
        if kind == 'error':
            self.close()
            raise value
        return value

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        while not self.queue.empty():
            self.queue.get_nowait()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def batches(dataset, split, *, prefetch=1, **settings):
    source = dataset.batches(split, **settings)
    if not prefetch:
        yield from source
    else:
        with PrefetchedBatches(source, prefetch) as stream:
            yield from stream


class _Array:
    """Placeholder for an array carried in a shared-memory slot."""
    __slots__ = ('offset', 'dtype', 'shape')

    def __init__(self, offset, dtype, shape):
        self.offset, self.dtype, self.shape = offset, dtype, shape

    def __reduce__(self):
        return _Array, (self.offset, self.dtype, self.shape)


# Arrays at least this large travel through shared memory; the rest of an
# update (targets, lists, small arrays) is pickled as before.
SHARED_ARRAY_BYTES = 1024


class _Tree:
    """A container holding shared arrays; containers without any are sent as they are."""
    __slots__ = ('kind', 'keys', 'items')

    def __init__(self, kind, keys, items):
        self.kind, self.keys, self.items = kind, keys, items

    def __reduce__(self):
        return _Tree, (self.kind, self.keys, self.items)


def _pack(value, arrays):
    """Skeleton of `value`: large arrays become slot placeholders at 64-byte offsets."""
    if isinstance(value, np.ndarray) and value.dtype != object and value.nbytes >= SHARED_ARRAY_BYTES:
        offset = arrays[-1][0]+-(-arrays[-1][1].nbytes//64)*64 if arrays else 0
        arrays.append((offset, value))
        return _Array(offset, value.dtype.str, value.shape)
    if isinstance(value, (dict, list, tuple)):
        keys = list(value) if isinstance(value, dict) else None
        items = [_pack(item, arrays) for item in (value.values() if keys is not None else value)]
        if any(isinstance(item, (_Array, _Tree)) for item in items):
            return _Tree(type(value).__name__, keys, items)
    return value


def _unpack(value, buffer):
    if isinstance(value, _Array):
        dtype = np.dtype(value.dtype)
        size = int(np.prod(value.shape, dtype=np.int64))*dtype.itemsize
        return buffer[value.offset:value.offset+size].view(dtype).reshape(value.shape)
    if isinstance(value, _Tree):
        items = [_unpack(item, buffer) for item in value.items]
        if value.kind == 'dict':
            return dict(zip(value.keys, items))
        return items if value.kind == 'list' else tuple(items)
    return value


class _Slots:
    """A reader's shared-memory slots; the learner returns each one after copying it."""
    def __init__(self, free, count):
        self.free, self.blocks = free, [None]*count
        for slot in range(count):
            free.put(slot)

    def write(self, slot, arrays):
        size = arrays[-1][0]+arrays[-1][1].nbytes if arrays else 0
        block = self.blocks[slot]
        if block is None or block.size < size:
            if block is not None:
                block.close(); block.unlink()
            # Grow with headroom so occasional large updates do not reallocate.
            block = self.blocks[slot] = shared_memory.SharedMemory(create=True, size=max(size*5//4, 1 << 24))
        for offset, array in arrays:
            target = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf, offset=offset)
            target[...] = array
            del target
        return block.name, size

    def close(self):
        for block in self.blocks:
            if block is not None:
                block.close(); block.unlink()
        self.blocks = []


# Decoded chunks each reader keeps (EventDataset.episode_rows): a lane reads each chunk of its episode once.
CACHE_BYTES = 256*1024*1024


def lane_parts(dataset, split, *, seed, epoch, lanes, steps, part_lanes, shuffle=True, cursor=0, share=None):
    """(update, part, its lanes, the update's schedule) for every part a rank prepares, from update `cursor`.

    An update is `lanes` lanes of `steps` consecutive decisions (EventDataset.lane_schedule),
    prepared in parts of `part_lanes` lanes: one part is about one memory-free batch of old
    (8 x 64 = 512 rows). With share=(rank, world), a rank takes every world-th part; the ranks'
    LSTMs then hold disjoint lanes, and losses are divided by the whole update's rows."""
    if part_lanes < 1 or lanes % part_lanes:
        raise ValueError('Lanes must be a whole number of parts')
    rank, world = share or (0, 1)
    parts = lanes//part_lanes
    if parts % world:
        raise ValueError('Parts must divide evenly between ranks')
    for update, schedule in enumerate(dataset.lane_schedule(split, seed=seed, epoch=epoch, lanes=lanes, steps=steps,
                                                            shuffle=shuffle)):
        if update < cursor:
            continue
        for part in range(rank, parts, world):
            yield update, part, range(part*part_lanes, (part+1)*part_lanes), schedule


def prepare_part(spec, dataset, split, item, pair_budget, cache_bytes):
    """One part's CPU arrays: its rows' microbatches with each row's lane and step, and the part's
    reset and present masks."""
    update, part, lanes, schedule = item
    rows = dataset.lane_rows(split, schedule, lanes, cache_bytes)
    prepared = prepare_update(spec, rows, pair_budget) if rows else {'rows': 0, 'microbatches': []}
    episode, _, reset = schedule
    prepared.update(update=update, part=part, first_lane=lanes.start, resets=reset[lanes.start:lanes.stop],
                    present=episode[lanes.start:lanes.stop] >= 0, update_rows=int((episode >= 0).sum()))
    return prepared


def assemble(parts, steps):
    """A rank's update from its parts in order: lanes renumbered locally, resets and present masks stacked.

    rows: this rank's rows; global_rows: the whole update's (the loss divisor)."""
    update = {'rows': sum(part['rows'] for part in parts), 'global_rows': parts[0]['update_rows'], 'steps': steps,
              'resets': np.concatenate([part['resets'] for part in parts]),
              'present': np.concatenate([part['present'] for part in parts]), 'microbatches': []}
    update['lanes'] = len(update['resets'])
    offset = 0
    for part in parts:
        for micro in part['microbatches']:
            micro['lanes'] = micro['lanes']-part['first_lane']+offset
            update['microbatches'].append(micro)
        offset += len(part['resets'])
    return update


def _serve(manifest, digest, spec_options, pair_budget, requests, results, free, slots, active, worker, workers,
           share=None):
    """Reader process `worker` of `workers`: of a rank's parts it prepares numbers worker, worker+workers, ...

    One long-lived dataset per process, so shards are verified once per run.
    Parts' arrays go through shared-memory slots, the skeleton through `results`.
    """
    from engine_runtime import disable_power_throttling
    from event_dataset import EventDataset
    serial = 0
    memory = None
    try:
        disable_power_throttling()
        dataset = EventDataset(manifest)
        if dataset.digest != digest:
            raise ValueError('Event dataset changed before the reader opened it')
        spec = PreparationSpec(dataset.config, dataset.key, **spec_options)
        memory = _Slots(free, slots)

        def send(item):
            while active.value == serial:
                try:
                    results.put(item, timeout=.1)
                    return True
                except Full:
                    pass
            return False

        def claim():
            while active.value == serial:
                try:
                    return free.get(timeout=.1)
                except Empty:
                    pass
            return None

        while (request := requests.get()) is not None:
            serial, split, settings = request
            cache = settings.pop('cache_bytes')
            for number, item in enumerate(lane_parts(dataset, split, share=share, **settings)):
                if number % workers != worker:
                    continue
                if active.value != serial:
                    break
                part = prepare_part(spec, dataset, split, item, pair_budget, cache)
                slot = claim()
                if slot is None:
                    break
                arrays = []
                skeleton = _pack(part, arrays)
                name, size = memory.write(slot, arrays)
                if not send((serial, 'item', (skeleton, slot, name, size))):
                    free.put(slot)
                    break
            else:
                send((serial, 'end', None))
    except BaseException:
        results.put((serial, 'error', traceback.format_exc()))
    finally:
        # The learner copies each slot before returning it, so none is in use.
        if memory is not None:
            memory.close()


class PreparedReader:
    """Prepared lane parts from reader processes, assembled into updates in the schedule's order.

    Reader w prepares a rank's parts w, w+processes, ... (lane_parts order), at most `depth`
    waiting each; the learner takes part k from reader k % processes. One stream is active at
    a time; abandoning one retires its serial, readers stop at their next part and leftovers
    are discarded by the following stream.
    """
    def __init__(self, dataset, spec, pair_budget, depth=1, processes=1, share=None):
        if depth < 1 or processes < 1:
            raise ValueError('Positive prefetch depth and reader count required')
        context = multiprocessing.get_context('spawn')
        self.dataset, self.share = dataset, share
        self.active, self.serial = context.Value('q', 0, lock=False), 0
        options = dict(float_dtype=spec.float_dtype, device_type=spec.device_type)
        self.readers, self.blocks = [], {}
        for worker in range(processes):
            requests, results, free = context.Queue(), context.Queue(depth), context.Queue()
            # Queued parts, one being written and one being copied here.
            process = context.Process(target=_serve, name=f'event-update-reader-{worker}', daemon=True,
                args=(str(dataset.path), dataset.digest, options, pair_budget,
                      requests, results, free, depth+2, self.active, worker, processes, share))
            process.start()
            self.readers.append((process, requests, results, free))

    def _receive(self, worker, value):
        """Copy a part out of its reader's slot, then return the slot."""
        skeleton, slot, name, size = value
        free = self.readers[worker][3]
        key = worker, slot
        if key not in self.blocks or self.blocks[key].name != name:
            if key in self.blocks:
                self.blocks[key].close()
            self.blocks[key] = shared_memory.SharedMemory(name=name, track=False)
        private = np.empty(size, dtype=np.uint8)  # Writable, like unpickled arrays.
        view = self.blocks[key].buf[:size]
        private[:] = np.frombuffer(view, dtype=np.uint8)
        view.release()
        free.put(slot)
        return _unpack(skeleton, private)

    def _next(self, worker, serial):
        process, _, results, free = self.readers[worker]
        while True:
            try:
                owner, kind, value = results.get(timeout=1)
            except Empty:
                if not process.is_alive():
                    # Negative: the signal that ended it (-11 a native crash, -9 killed from outside).
                    raise RuntimeError(f'Event update reader exited without a result (exit code {process.exitcode})') from None
                continue
            if kind == 'error':
                raise RuntimeError('Event update reader failed:\n'+value)
            if owner == serial:
                return kind, self._receive(worker, value) if kind == 'item' else value
            if kind == 'item':
                free.put(value[1])  # Abandoned stream: discard and recycle its slot.

    def stream(self, split, **settings):
        """(update, following cursor) pairs in the schedule's order."""
        if not self.readers:
            raise RuntimeError('Event update reader is closed')
        self.serial += 1
        serial = self.serial
        self.active.value = serial
        settings = {'cache_bytes': CACHE_BYTES, **settings}
        for _, requests, _, _ in self.readers:
            requests.put((serial, split, dict(settings)))
        rank, world = self.share or (0, 1)
        count = settings['lanes']//settings['part_lanes']//world
        number, parts = 0, []
        try:
            while True:
                kind, value = self._next(number % len(self.readers), serial)
                if kind == 'end':
                    return
                number += 1
                parts.append(value)
                if len(parts) == count:
                    yield assemble(parts, settings['steps']), parts[0]['update']+1
                    parts = []
        finally:
            if self.active.value == serial:
                self.active.value = 0

    def close(self):
        if not self.readers:
            return
        self.active.value = 0
        for _, requests, _, _ in self.readers:
            requests.put(None)
        # Drain so blocked puts observe the retired serial and the readers exit.
        for process, _, results, _ in self.readers:
            while process.is_alive():
                try:
                    results.get(timeout=.1)
                except Empty:
                    pass
            process.join()
        for block in self.blocks.values():
            block.close()
        self.readers, self.blocks = [], {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def prepared(dataset, split, *, spec, pair_budget, reader=None, share=None, cache_bytes=None, **settings):
    """(update, following cursor) from `reader` (built with the same share), or prepared inline when it is None.

    settings: seed, epoch, lanes, steps, part_lanes, shuffle and cursor (lane_parts)."""
    cache_bytes = cache_bytes or CACHE_BYTES
    if reader is not None:
        yield from reader.stream(split, cache_bytes=cache_bytes, **settings)
        return
    rank, world = share or (0, 1)
    count = settings['lanes']//settings['part_lanes']//world
    parts = []
    for item in lane_parts(dataset, split, share=share, **settings):
        parts.append(prepare_part(spec, dataset, split, item, pair_budget, cache_bytes))
        if len(parts) == count:
            yield assemble(parts, settings['steps']), item[0]+1
            parts = []
