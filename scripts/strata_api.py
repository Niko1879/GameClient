"""Generals Online replay list from the Strata API (replaces crawling the match listing for discovery).

API: GET https://strata.gamereplays.org/api/replays (spec: /docs/api.json), bearer token in .secrets/generals-online.key
(gitignored; never logged). Returns only match_id + replay_url, newest first, cursor-paged (per_page <= 500); the
history starts mid-July 2026. Filters used here: match_types[]=1v1, any China side (3 China, 8 Tank, 10 Nuke), min_elo (default 1351, the top-20% floor).
Players, ratings and results are NOT in the API; they still come from the replay file and the listing/match pages.

usage: python -B scripts/strata_api.py list [--from ISO] [--to ISO] [--sides 3 8 10] [--min-elo 1351] [--delay 1]
  writes replays/catalog/strata-list.jsonl ({match_id, replay_url, listed_utc}), appending; resumable via its cursor file.
       python -B scripts/strata_api.py inject
  gives listed matches that the crawler queues hold as failed/pending an api_replay_url and makes them due now.
"""
import argparse, json, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT/'.secrets/generals-online.key'
API = 'https://strata.gamereplays.org/api/replays'
LIST = ROOT/'replays/catalog/strata-list.jsonl'
UA = 'GeneralsAIReplayCollector/1.0'


def request(params, attempts=5):
    url = API+'?'+urllib.parse.urlencode(params, doseq=True)
    headers = {'Authorization': 'Bearer '+KEY.read_text().strip(), 'Accept': 'application/json', 'User-Agent': UA}
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 422):
                raise SystemExit(f'API refused ({error.code}): {error.read()[:300]!r}')
            wait = int(error.headers.get('Retry-After') or 0) or 30*(attempt+1)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            wait = 30*(attempt+1)
        time.sleep(wait)
    raise SystemExit('API unreachable after retries')


def list_matches(args):
    state = LIST.with_suffix('.cursor.json')
    query = {'match_types[]': ['1v1'], 'sides[]': args.sides, 'side_operator': 'or', 'per_page': 500,
             **({'min_elo': args.min_elo} if args.min_elo else {}), **({'from': args.start} if args.start else {}), **({'to': args.end} if args.end else {})}
    saved = json.loads(state.read_text()) if state.exists() else {}
    cursor = saved.get('cursor') if saved.get('query') == query else None
    LIST.parent.mkdir(parents=True, exist_ok=True)
    pages = rows = 0
    with LIST.open('a', encoding='utf-8') as out:
        while True:
            body = request({**query, **({'cursor': cursor} if cursor else {})})
            now = datetime.now(timezone.utc).isoformat(timespec='seconds')
            for item in body['data'] or []:
                out.write(json.dumps({'match_id': item['match_id'], 'replay_url': item['replay_url'], 'listed_utc': now})+'\n')
            out.flush()
            rows += len(body['data'] or []); pages += 1
            cursor = body['next_cursor']
            state.write_text(json.dumps({'query': query, 'cursor': cursor}))
            if pages % 50 == 0:
                print(f'{pages} pages, {rows} matches', flush=True)
            if not cursor:
                break
            time.sleep(args.delay)
    state.unlink()
    print(f'done: {pages} pages, {rows} matches -> {LIST}')


def inject(args):
    """Hand the API links to the crawler queues: failed/pending matches get api_replay_url and are due now.

    The crawler (harvest_china_replays.Archive.harvest) tries the page link, then this, then the listing link, so
    files, sidecars, checks and the catalogue sync stay the crawler's. Matches the crawler hasn't discovered yet are
    left to it (it needs the listing row: factions, text).
    """
    import sqlite3
    links = {}
    with LIST.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            links[row['match_id']] = row['replay_url']
    for archive in args.archives:
        db = sqlite3.connect(ROOT/archive/'harvest.sqlite3', timeout=60)
        counts = {}
        rows = db.execute("SELECT id,candidate,status FROM matches WHERE status IN ('error','pending')").fetchall()
        for match_id, candidate, status in rows:
            if match_id not in links:
                continue
            candidate = json.loads(candidate)
            if candidate.get('api_replay_url') == links[match_id] and status == 'pending':
                continue
            candidate['api_replay_url'] = links[match_id]
            db.execute("UPDATE matches SET candidate=?,status='pending',retry_after=NULL WHERE id=?",
                       (json.dumps(candidate), match_id))
            counts[status] = counts.get(status, 0)+1
        db.commit(); db.close()
        print(archive, 'queued with API links:', counts)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    queue = sub.add_parser('inject')
    queue.add_argument('--archives', nargs='+', default=['replays/vanilla-china', 'replays/china-generals'])
    listing = sub.add_parser('list')
    listing.add_argument('--from', dest='start')
    listing.add_argument('--to', dest='end')
    listing.add_argument('--sides', type=int, nargs='+', default=[3, 8, 10])
    listing.add_argument('--min-elo', type=int, default=1351)
    listing.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    if args.command == 'list':
        list_matches(args)
    elif args.command == 'inject':
        inject(args)


if __name__ == '__main__':
    main()
