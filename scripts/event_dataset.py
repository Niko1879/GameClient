"""Prepared event datasets: verified shard reads and the deterministic training sampler.

Reading and sampling never change prepared data, so this module is outside the
feature identity (event_store.FEATURE_SOURCES) and can change without invalidating
the corpus.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from event_store import SCHEMA
from generals_policy import ARCHITECTURE, OPENER_WINDOW_SECONDS, UNRATED, opener_inputs, rating_inputs
from openers import blanked
from prepared_io import COMMON_FIELDS, _digest, _json_bytes
from tensor_replay_store import ARRAY_FIELDS, LIST_FIELDS, MASK_FIELDS, decode_columns


class _Views:
    """A decoded column's ragged index as Python lists: one conversion per chunk."""
    __slots__ = ('data', 'ids', 'offsets', 'shapes', 'orders')

    def __init__(self, column):
        self.data, self.ids, self.offsets = column.data, column.ids.tolist(), column.offsets.tolist()
        self.shapes = [tuple(shape) for shape in column.shapes.tolist()]
        self.orders = ['F' if order else 'C' for order in column.orders.tolist()]

    def at(self, row):
        index = self.ids[row]
        if index < 0:
            return None
        return self.data[self.offsets[index]:self.offsets[index+1]].reshape(self.shapes[index], order=self.orders[index])


def chunk_rows(columns, metadata, common, key):
    """Every row of a decoded chunk, equal to tensor_replay_store.TensorRows' views.

    TensorRows is a feature source (editing it changes the feature key), so the
    training reader builds the same rows here in one pass over the chunk.
    test_event_dataset pins the two to identical rows on stored shards.
    """
    views = {name: _Views(column) for name, column in columns.items()}
    encoded_fields = [(name, views['e/'+name]) for name in ARRAY_FIELDS if 'e/'+name in views]
    mask_fields = [(name, views['l/'+name]) for name in MASK_FIELDS if 'l/'+name in views]
    list_fields = [(name, views['l/'+name]) for name in LIST_FIELDS if 'l/'+name in views]
    variants = {}
    for name in columns:
        if '|' in name:
            variants.setdefault(name.partition('|')[0], []).append(views[name])
    result = []
    for item, meta in enumerate(metadata):
        encoded = {}
        for name, view in encoded_fields:
            value = view.at(item)
            if value is not None:
                encoded[name] = value
        chosen = {}
        for name, candidates in variants.items():
            values = [value for view in candidates if (value := view.at(item)) is not None]
            if len(values) > 1:
                raise ValueError('Overlapping training column variants')
            if values:
                chosen[name] = values[0]
        encoded.update({name[2:]: value for name, value in chosen.items() if name.startswith('e/')})
        encoded.update(meta['encoded'])
        for name in COMMON_FIELDS:
            if name not in encoded and name in common:
                encoded[name] = common[name]
        target = meta['row']['target']
        mask = {'key': key, 'teacher': target}
        mask.update({name[2:]: value for name, value in chosen.items() if name.startswith('l/')})
        for name, view in mask_fields:
            value = view.at(item)
            if value is not None:
                mask[name] = value
        for name, view in list_fields:
            value = view.at(item)
            if value is not None:
                mask[name] = value.tolist()
        encoded['_training_legality'] = mask
        result.append({**meta['row'], 'encoded': encoded})
    return result


def player_rating_inputs(ratings):
    """{player index: model rating inputs}: own rating first, then the other player's."""
    return {player: rating_inputs(rating, next((other for key, other in ratings.items() if key != player), None))
            for player, rating in ratings.items()}


# Games' shared map data kept decoded: sequence training reads one game per lane at a time.
SHARED_GAMES = 1024


class EventDataset:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.digest = _digest(self.path)
        self.document = json.loads(self.path.read_text())
        if (self.document.get('kind') != 'generals_event_dataset' or self.document.get('schema') != SCHEMA
                or self.document.get('complete') is False):
            raise ValueError('Unsupported event dataset')
        self.config = self.document['config']
        if self.config.get('architecture') != ARCHITECTURE:
            raise ValueError('Not an event policy dataset')
        seen, keys = set(), set()
        for entry in self.document['entries']:
            if hashlib.sha256(_json_bytes(entry['index'])).hexdigest() != entry['index_sha256']:
                raise ValueError('Event dataset index proof mismatch')
            match = entry['match_key']
            if match in seen or entry['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Duplicate match or invalid whole-match split')
            seen.add(match); keys.add(entry['index']['legality_key'])
        if len(keys) != 1:
            raise ValueError('Event dataset mixes feature identities')
        self.key = next(iter(keys))
        self.entries = self.document['entries']
        self.verified, self.shared = {}, OrderedDict()
        self._episodes, self._chunks = {}, OrderedDict()

    def shard_path(self, entry):
        path = (self.path.parent / entry['path']).resolve()
        if self.path.parent not in path.parents:
            raise ValueError('Event shard path escapes dataset directory')
        return path

    def chunks(self, split):
        for source, entry in enumerate(self.entries):
            if entry['split'] != split:
                continue
            # Players below the selection's rating floor (run_prepared_training.selection, 17 Nov 2026): the floor is
            # per player, so a weaker China player in a China-vs-China match is not supervised.
            excluded = set(entry.get('excluded_players', ()))
            for episode in entry['index']['episodes']:
                if str(episode['player']) in excluded:
                    continue
                for descriptor in episode['chunks']:
                    yield source, episode['shared'], descriptor

    def read(self, reference):
        source, shared, descriptor = reference
        entry = self.entries[source]
        path = self.shard_path(entry)
        stamp = path.stat().st_size, path.stat().st_mtime_ns
        if source not in self.verified:
            if _digest(path) != entry['archive_sha256']:
                raise ValueError('Event shard integrity mismatch')
            self.verified[source] = stamp
        elif self.verified[source] != stamp:
            raise ValueError('Event shard changed while training')

        def read_member(archive, item):
            if archive.getinfo(item['member']).file_size != item['bytes']:
                raise ValueError('Event column size mismatch')
            raw = archive.read(item['member'])
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Event column hash mismatch')
            return decode_columns(raw)

        with zipfile.ZipFile(path) as archive:
            identity = source, shared
            if identity not in self.shared:
                columns, meta = read_member(archive, entry['index']['shared'][shared])
                common = {name: column.at(0) for name, column in columns.items()}
                common.update(meta[0])
                self.shared[identity] = common
                # Immutable map working sets, one per game a sequence lane may be reading.
                while len(self.shared) > SHARED_GAMES:
                    self.shared.popitem(last=False)
            self.shared.move_to_end(identity)
            columns, metadata = read_member(archive, descriptor)
        if len(metadata) != descriptor['rows']:
            raise ValueError('Event column row count mismatch')
        rows = chunk_rows(columns, metadata, self.shared[identity], self.key)
        # Not stored in the shard: run_prepared_training.selection adds both players' identities from the
        # replay; a row's enemy is the one that is not its own (mixed vanilla/Tank/Nuke matches, 13 Nov 2026).
        identities = [tuple(i) for i in (entry.get('faction_identities') or ())]
        # Pre-match ratings by player index (None: unrated), set by run_prepared_training.selection
        # like enemy_identity; each row is one player's view.
        inputs = player_rating_inputs(entry.get('ratings', {}))
        # Strategy input z (scripts/openers.py), set by run_prepared_training.selection per shard player index;
        # absent means no opener for anyone, like ratings (cloud_bundle refuses a selection without any).
        openers = entry.get('openers', {})
        for row in rows:
            own = tuple(row['encoded']['player_identity'])
            row['encoded']['enemy_identity'] = (identities[1] if own == identities[0] else identities[0])                 if len(identities) == 2 and own in identities else (0, 0)
            row['encoded']['ratings'] = inputs.get(str(row['player']), UNRATED)
            opener = openers.get(str(row['player']))
            seconds = row['frame']/row.get('source_ticks_per_second', 60)
            row['opener'] = ('none' if opener is None else 'after' if seconds >= OPENER_WINDOW_SECONDS
                             else 'blanked' if blanked(entry['match_key'], row['player']) else 'given')
            row['encoded']['opener'] = opener_inputs(self.config, opener['templates'] if row['opener'] == 'given' else None)
            # The acting player's pre-match rating (estimated or not), for reporting only (train_events by_rating).
            rating = entry.get('ratings', {}).get(str(row['player']))
            row['rating'] = rating['rating'] if isinstance(rating, dict) else rating
            # Reporting only: names the match of an extreme-loss decision (train_events outliers).
            row['match_key'] = entry['match_key']
        return rows

    def windows(self, split, *, seed, epoch, buffer_bytes=64*1024*1024):
        """Shuffled chunk references grouped into byte-bounded windows.

        Windows are independent: each has its own deterministic RNG, so readers
        may decode different windows concurrently without changing any batch.
        """
        if buffer_bytes < 1:
            raise ValueError('Positive shuffle-buffer budget required')
        chunks = list(self.chunks(split))
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(chunks))
        windows, current, size = [], [], 0
        for ordinal in order:
            reference = chunks[int(ordinal)]
            estimate = reference[2]['bytes']*3
            if current and size+estimate > buffer_bytes:
                windows.append(current); current, size = [], 0
            current.append(reference); size += estimate
        if current:
            windows.append(current)
        return windows

    def window_batches(self, references, *, seed, epoch, window, batch_size=256, start=0):
        """One window's batches, as (rows, position after this batch), from `start`."""
        if batch_size < 1:
            raise ValueError('Positive batch budget required')
        rows = []
        for reference in references:
            rows.extend(self.read(reference))
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, window, 1]))
        rng.shuffle(rows)
        # Stable sorting keeps random order inside entity-count buckets.
        rows.sort(key=lambda row: len(row['encoded']['numeric'])//16)
        batches = [rows[first:first+batch_size] for first in range(0, len(rows), batch_size)]
        positions = rng.permutation(len(batches))
        for position in range(start, len(positions)):
            yield batches[int(positions[position])], position+1

    def batches(self, split, *, seed, epoch, batch_size=256, buffer_bytes=64*1024*1024,
                cursor=(0, 0)):
        """Shuffle chunks, then bucket a byte-bounded window by entity count.

        Cursor is (window, batch). Rebuilding it needs neither earlier decoded
        data nor neural state. All windows use their own deterministic RNG.
        """
        if batch_size < 1 or buffer_bytes < 1:
            raise ValueError('Positive batch and shuffle-buffer budgets required')
        windows = self.windows(split, seed=seed, epoch=epoch, buffer_bytes=buffer_bytes)
        for window in range(cursor[0], len(windows)):
            start = cursor[1] if window == cursor[0] else 0
            for rows, position in self.window_batches(windows[window], seed=seed, epoch=epoch, window=window,
                                                      batch_size=batch_size, start=start):
                yield rows, (window, position)

    # Sequence training (notes/plan.md Â§2): lanes of consecutive decisions, one episode at a time.

    def episodes(self, split):
        """The split's training units in index order: (chunk references in game order, rows).

        An episode is one supervised player's decisions in one match; below-floor players'
        episodes stay skipped, as in chunks()."""
        cached = self._episodes.get(split)
        if cached is None:
            cached = []
            for source, entry in enumerate(self.entries):
                if entry['split'] != split:
                    continue
                excluded = set(entry.get('excluded_players', ()))
                for episode in entry['index']['episodes']:
                    if str(episode['player']) in excluded:
                        continue
                    chunks = [(source, episode['shared'], descriptor) for descriptor in episode['chunks']]
                    rows = sum(descriptor['rows'] for descriptor in episode['chunks'])
                    if rows:
                        cached.append((chunks, rows))
            self._episodes[split] = cached
        return cached

    def lane_schedule(self, split, *, seed, epoch, lanes, steps, shuffle=True):
        """Every update of an epoch as int arrays (lanes, steps): (episode or -1 for no row, row
        within the episode, reset = an episode's first decision).

        Each lane plays one episode from its first decision to its last, then takes the next
        episode of a seeded permutation (index order with shuffle=False), so an update holds
        `lanes` different games at different phases. Lanes drain at the epoch's end; the last
        updates are partly empty. A pure function of the index: a cursor is an update number."""
        if lanes < 1 or steps < 1:
            raise ValueError('Positive lane and step counts required')
        episodes = self.episodes(split)
        order = (np.random.default_rng(np.random.SeedSequence([seed, epoch, 2])).permutation(len(episodes))
                 if shuffle else np.arange(len(episodes)))
        queue = iter(order.tolist())
        current = [None]*lanes
        while True:
            episode = np.full((lanes, steps), -1, dtype=np.int64)
            row, reset = np.zeros((lanes, steps), dtype=np.int64), np.zeros((lanes, steps), dtype=bool)
            for lane in range(lanes):
                step = 0
                while step < steps:
                    if current[lane] is None:
                        following = next(queue, None)
                        if following is None:
                            break
                        current[lane] = (following, 0)
                    number, start = current[lane]
                    take = min(steps-step, episodes[number][1]-start)
                    episode[lane, step:step+take] = number
                    row[lane, step:step+take] = np.arange(start, start+take)
                    reset[lane, step] = start == 0
                    step += take
                    current[lane] = None if start+take == episodes[number][1] else (number, start+take)
            if (episode < 0).all():
                return
            yield episode, row, reset

    def episode_rows(self, split, number, start, stop, cache_bytes=256*1024*1024):
        """Rows [start, stop) of one episode, in game order.

        Decoded chunks stay in a least-recently-used cache of about cache_bytes (three times their
        stored size each), so a lane reading its episode across updates decodes each chunk once."""
        chunks, _ = self.episodes(split)[number]
        rows, offset = [], 0
        for reference in chunks:
            count = reference[2]['rows']
            if offset < stop and offset+count > start:
                key = reference[0], reference[2]['member']
                cached = self._chunks.get(key)
                if cached is None:
                    cached = self._chunks[key] = (self.read(reference), reference[2]['bytes']*3)
                    while len(self._chunks) > 1 and sum(item[1] for item in self._chunks.values()) > cache_bytes:
                        self._chunks.popitem(last=False)
                self._chunks.move_to_end(key)
                rows.extend(cached[0][max(0, start-offset):stop-offset])
            offset += count
        return rows

    def lane_rows(self, split, schedule, lanes, cache_bytes=256*1024*1024):
        """One update's rows of the given lanes, each a shallow copy with its 'lane' and 'step'."""
        episode, row, _ = schedule
        result = []
        for lane in lanes:
            numbers = episode[lane]
            step = 0
            while step < len(numbers) and numbers[step] >= 0:
                end = step
                while end < len(numbers) and numbers[end] == numbers[step]:
                    end += 1
                values = self.episode_rows(split, int(numbers[step]), int(row[lane, step]), int(row[lane, end-1])+1,
                                           cache_bytes)
                if len(values) != end-step:
                    raise ValueError('Episode rows do not match the shard index')
                result.extend({**value, 'lane': lane, 'step': position} for position, value in enumerate(values, start=step))
                step = end
        return result
