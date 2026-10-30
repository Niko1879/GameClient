"""Bounded prefetch that preserves the event sampler's order and cursors.

`PrefetchedBatches` decodes rows on a thread. `PreparedReader` runs a separate
process (no PyTorch) that decodes shards and builds each update's CPU arrays:
microbatches, padded inputs and teacher plans. The learner process then only
stages and runs them, so CPU preparation overlaps GPU work without contending
for the learner's interpreter lock. Readers change speed only, never content.
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


def _serve(manifest, digest, spec_options, pair_budget, requests, results, free, slots, active, worker, workers):
    """Reader process `worker` of `workers`: it prepares windows worker, worker+workers, ...

    One long-lived dataset per process, so shards are verified once per run.
    Updates' arrays go through shared-memory slots, the skeleton through `results`.
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

        def send_update(update, cursor):
            slot = claim()
            if slot is None:
                return False
            arrays = []
            skeleton = _pack(update, arrays)
            name, size = memory.write(slot, arrays)
            if send((serial, 'item', (skeleton, cursor, slot, name, size))):
                return True
            free.put(slot)
            return False

        while (request := requests.get()) is not None:
            serial, split, settings = request
            cursor = settings.pop('cursor', (0, 0))
            windows = dataset.windows(split, seed=settings['seed'], epoch=settings['epoch'],
                                      buffer_bytes=settings.get('buffer_bytes', 64*1024*1024))
            for window in range(cursor[0], len(windows)):
                if window % workers != worker:
                    continue
                start = cursor[1] if window == cursor[0] else 0
                for rows, position in dataset.window_batches(windows[window], seed=settings['seed'],
                        epoch=settings['epoch'], window=window, batch_size=settings.get('batch_size', 256), start=start):
                    if active.value != serial or not send_update(prepare_update(spec, rows, pair_budget),
                                                                 (window, position)):
                        break
                if active.value != serial or not send((serial, 'window', window)):
                    break
    except BaseException:
        results.put((serial, 'error', traceback.format_exc()))
    finally:
        # The learner copies each slot before returning it, so none is in use.
        if memory is not None:
            memory.close()


class PreparedReader:
    """Prepared updates from reader processes, in exactly the sampler's order.

    Shuffle windows are independent (event_dataset.EventDataset.windows), so
    `processes` readers take windows in turn and the learner consumes window w
    from reader w % processes. At most `depth` updates wait per reader. One
    stream is active at a time; abandoning one retires its serial, readers stop
    at their next update and leftovers are discarded by the following stream.
    """
    def __init__(self, dataset, spec, pair_budget, depth=1, processes=1):
        if depth < 1 or processes < 1:
            raise ValueError('Positive prefetch depth and reader count required')
        context = multiprocessing.get_context('spawn')
        self.dataset = dataset
        self.active, self.serial = context.Value('q', 0, lock=False), 0
        options = dict(float_dtype=spec.float_dtype, device_type=spec.device_type)
        self.readers, self.blocks = [], {}
        for worker in range(processes):
            requests, results, free = context.Queue(), context.Queue(depth), context.Queue()
            # Queued updates, one being written and one being copied here.
            process = context.Process(target=_serve, name=f'event-update-reader-{worker}', daemon=True,
                args=(str(dataset.path), dataset.digest, options, pair_budget,
                      requests, results, free, depth+2, self.active, worker, processes))
            process.start()
            self.readers.append((process, requests, results, free))

    def _receive(self, worker, value):
        """Copy an update out of its reader's slot, then return the slot."""
        skeleton, cursor, slot, name, size = value
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
        return _unpack(skeleton, private), cursor

    def _next(self, worker, serial):
        process, _, results, free = self.readers[worker]
        while True:
            try:
                owner, kind, value = results.get(timeout=1)
            except Empty:
                if not process.is_alive():
                    raise RuntimeError('Event update reader exited without a result') from None
                continue
            if kind == 'error':
                raise RuntimeError('Event update reader failed:\n'+value)
            if owner == serial:
                return kind, self._receive(worker, value) if kind == 'item' else value
            if kind == 'item':
                free.put(value[2])  # Abandoned stream: discard and recycle its slot.

    def stream(self, split, **settings):
        """(prepared update, following cursor) pairs in the sampler's order."""
        if not self.readers:
            raise RuntimeError('Event update reader is closed')
        self.serial += 1
        serial = self.serial
        self.active.value = serial
        cursor = tuple(settings.get('cursor', (0, 0)))
        windows = len(self.dataset.windows(split, seed=settings['seed'], epoch=settings['epoch'],
                                           buffer_bytes=settings.get('buffer_bytes', 64*1024*1024)))
        for _, requests, _, _ in self.readers:
            requests.put((serial, split, dict(settings)))
        try:
            for window in range(cursor[0], windows):
                worker = window % len(self.readers)
                while True:
                    kind, value = self._next(worker, serial)
                    if kind == 'window':
                        if value != window:
                            raise RuntimeError('Event update readers lost window order')
                        break
                    yield value
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


def prepared(dataset, split, *, spec, pair_budget, reader=None, **settings):
    """Prepared updates from `reader`, or prepared inline when it is None."""
    if reader is not None:
        yield from reader.stream(split, **settings)
        return
    for rows, cursor in dataset.batches(split, **settings):
        yield prepare_update(spec, rows, pair_budget), cursor
