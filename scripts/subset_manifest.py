"""Experiment manifests over an uploaded bundle's shards: no new shards, no upload.

A subset manifest keeps the source dataset's validation entries unchanged (one
yardstick for every experiment) and filters its training entries. It is written
next to the source manifest, because shard paths resolve against the manifest's
own directory.

  # locally, once: the curated gamereplays.org match keys (catalogue read)
  python -B scripts/subset_manifest.py curated-keys --output curated-keys.json
  # on the instance
  python -B code/subset_manifest.py subset --source data/dataset.json \\
      --output data/curated.json --min-rating 1400 --include curated-keys.json
  # a rung of the rating ladder. The user's rungs are shares of ALL replays (GO + GenTool + gamereplays), not of
  # the bundle, so use the fixed cuts from 12 Nov: top 10% = --min-rating 1574, top 20% = --min-rating 1356.
  # --top-fraction ranks only the bundle's own training matches; use it only on a bundle of the whole pool.
  python -B code/subset_manifest.py subset --source data/dataset.json \\
      --output data/top10.json --min-rating 1574
  python -B code/subset_manifest.py subset --source data/dataset.json \\
      --output data/wide.json --widths '{"map_hidden": 64}'
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from event_store import MODEL_ONLY


def match_rating(entry):
    """The match's rating for the ladder: the highest pre-match rating among the players whose play
    the shard holds (its China players), None when none has one.

    The same rule the export queue is ordered by (highest China rating first). A rating is
    replay_cleaning.player_ratings' {'rating': value, 'estimated': bool}; estimates count.
    The opponent's rating says nothing about the demonstrations, so it is not used.
    """
    ratings = entry.get('ratings', {})
    values = [ratings[player]['rating'] for player in {str(episode['player']) for episode in entry['index']['episodes']}
              if ratings.get(player) is not None and ratings[player]['rating'] is not None]
    return max(values) if values else None


def top_fraction_rating(document, fraction):
    """The rating at or above which the top `fraction` of rated training matches lie.

    Cuts are nested: a larger fraction keeps every match a smaller one keeps."""
    if not 0 < fraction <= 1:
        raise ValueError('The top fraction must be in (0, 1]')
    ratings = sorted((rating for entry in document['entries'] if entry['split'] == 'train'
                      and (rating := match_rating(entry)) is not None), reverse=True)
    if not ratings:
        raise ValueError('No rated training matches')
    return ratings[max(1, round(fraction*len(ratings)))-1]


def subset(document, *, min_rating=None, top_fraction=None, include=(), widths=None):
    """A new dataset document: every validation entry, the training entries that pass."""
    include = set(include)
    if widths and not set(widths) <= MODEL_ONLY:
        raise ValueError('Only model widths may change: '+', '.join(sorted(set(widths)-MODEL_ONLY)))
    if top_fraction is not None:
        if min_rating is not None:
            raise ValueError('Give a minimum rating or a top fraction, not both')
        min_rating = top_fraction_rating(document, top_fraction)
    kept, counts = [], {'validation': 0, 'rated': 0, 'included': 0}
    for entry in document['entries']:
        if entry['split'] != 'train':
            kept.append(entry); counts['validation'] += 1
            continue
        rating = match_rating(entry)
        if min_rating is None or (rating is not None and rating >= min_rating):
            kept.append(entry); counts['rated'] += 1
        elif entry['match_key'] in include:
            kept.append(entry); counts['included'] += 1
    result = {**document, 'entries': kept, 'config': {**document['config'], **(widths or {})}}
    result['selection'] = {**document.get('selection', {}),
                           'subset': dict(min_rating=min_rating, top_fraction=top_fraction, include_sha256=hashlib.sha256(
                               json.dumps(sorted(include)).encode()).hexdigest(), widths=widths or {}, counts=counts)}
    return result


def curated_keys(catalog):
    """Match keys of the curated gamereplays.org archive (run_prepared_training.ARCHIVE_SOURCES)."""
    import sqlite3
    from run_prepared_training import ARCHIVE_SOURCES
    db = sqlite3.connect(f'file:{Path(catalog).as_posix()}?mode=ro', uri=True, timeout=30)
    try:
        rows = db.execute('SELECT DISTINCT match_key FROM conversion_job WHERE instr(source_path,?)>0',
                          (ARCHIVE_SOURCES['curated'],)).fetchall()
    finally:
        db.close()
    return sorted(key for (key,) in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    keys = sub.add_parser('curated-keys')
    keys.add_argument('--catalog', default='replays/catalog/replay-catalog.sqlite3')
    keys.add_argument('--output', required=True)
    make = sub.add_parser('subset')
    make.add_argument('--source', required=True)
    make.add_argument('--output', required=True)
    make.add_argument('--min-rating', type=float,
                      help="Training matches whose highest China player's rating is at least this")
    make.add_argument('--top-fraction', type=float,
                      help='Instead: the top share of training matches by that rating, e.g. 0.1 (nested cuts)')
    make.add_argument('--include', help='JSON list of match keys kept in training regardless of rating')
    make.add_argument('--widths', help='JSON object of model-only widths')
    args = parser.parse_args()
    if args.command == 'curated-keys':
        found = curated_keys(args.catalog)
        Path(args.output).write_text(json.dumps(found))
        print(json.dumps({'curated_keys': len(found)}))
        return
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.parent != source.parent or output == source:
        raise ValueError('Write the subset next to (not over) its source: shard paths are relative to it')
    document = json.loads(source.read_text())
    result = subset(document, min_rating=args.min_rating, top_fraction=args.top_fraction,
                    include=json.loads(Path(args.include).read_text()) if args.include else (),
                    widths=json.loads(args.widths) if args.widths else None)
    output.write_text(json.dumps(result))
    print(json.dumps(result['selection']['subset']))


if __name__ == '__main__':
    main()
