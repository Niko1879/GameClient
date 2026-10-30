"""Refresh the local replay catalogue and materialise prepared selections.

Reads the crawler archive read-only and re-registers prepared shards beside the
training corpus. It never edits training inputs or source pins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay_catalog import (connect as durable_connect, build_dataset, import_prepared_directory,
                            import_test_reserve, refresh_availability, DATA)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "replays/vanilla-china/harvest.sqlite3"
CATALOG = ROOT / "replays/catalog/replay-catalog.sqlite3"


def connect(path: Path):
    db = durable_connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    return db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-only', action='store_true', help='Skip refreshing the raw crawler ledger')
    parser.add_argument('--list-features', action='store_true')
    parser.add_argument('--make-dataset', type=Path)
    parser.add_argument('--feature-key')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--split', choices=('train','validation','test','all'), default='train')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    db = connect(CATALOG)
    try:
        if args.list_features:
            print(json.dumps(db.execute('SELECT feature_key,count(*) FROM prepared_shard WHERE available=1 GROUP BY feature_key').fetchall(),indent=2))
            return
        if args.make_dataset:
            if not args.feature_key or (args.limit is not None and args.limit<=0):
                parser.error('Dataset selection requires --feature-key and a positive --limit when specified')
            output = args.make_dataset.resolve()
            if output.parent != DATA.resolve() or output.exists():
                parser.error('Choose a new manifest filename directly beside the stored training shards')
            doc = build_dataset(db,args.feature_key,limit=args.limit,split=args.split,seed=args.seed)
            output.write_text(json.dumps(doc,separators=(',',':'),sort_keys=True)+'\n',encoding='utf-8')
            print(json.dumps({'manifest':str(output),'replays':len(doc['entries'])}))
            return
        if not args.catalog_only:
            from replay_conversion import refresh_collection
            refresh_collection(db, ARCHIVE)
        prepared_import = import_prepared_directory(db,DATA)
        reserved = import_test_reserve(db,ROOT)
        refresh_availability(db)
        raw_counts = dict(db.execute("SELECT raw_status,count(*) FROM replay GROUP BY raw_status")) \
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='replay'").fetchone() else {}
        print(json.dumps({"catalog": str(CATALOG),
                          "raw_replays_by_status": raw_counts,
                          "prepared_import": prepared_import,
                          "test_reserve": reserved,
                          "durable_export_results": db.execute('SELECT count(*) FROM export_result').fetchone()[0],
                          "available_prepared_shards": db.execute('SELECT count(*) FROM prepared_shard WHERE available=1').fetchone()[0]},
                         indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
