"""Record who won each Generals Online match, from strata.gamereplays.org (16 Nov 2026).

Replays carry no result, so the site is the source:
- Rated matches: the listing row the crawler already saved (sidecar `listing_text`) shows each player's rating
  change ("+15 Seasonal: 1110 -> 1125 ..."); the player whose rating rose won. No request needed.
- Other matches (stats disabled, unrated lobbies): the match page has a per-player Result column (Won/Lost) and
  per-player stats. `--fetch` downloads it through the crawler's client (robots.txt, user agent, per-host pacing).

Output: replays/catalog/match-results.jsonl, one line per match, appended and resumable:
  {"match_key": "pgo:<id>", "match_id": <id>, "source": "listing"|"page", "players": [{"slot": <header slot or
   null>, "name": ..., "result": "won"|"lost"|<page text>, ...}], "recorded_utc": ...}
Slots are the replay header's slot numbers (cleaning metrics' `slot`; player_index = slot + 2 there).
Matches whose page shows no result are recorded with source "page" and an empty `players` list, so they are
not fetched again.

usage:
  python -B scripts/match_results.py backfill [--fetch] [--limit N] [--delay 5] [--keys FILE]
  python -B scripts/match_results.py verify [--sample 30]
  python -B scripts/match_results.py summary
"""
import argparse
import datetime
import json
import os
import random
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
OUT = ROOT/'replays/catalog/match-results.jsonl'
FOLDERS = ('vanilla-china', 'vanilla-china-td', 'china-generals')  # crawler archives (GO matches with sidecars)
DELTA = r'(?P<d{k}>[+-]\d+) (?P<b{k}>[A-Za-z][A-Za-z ]*?): (?P<a{k}>\d+) → (?P<c{k}>\d+)'
STATS = ('Units Created', 'Units Lost', 'Units Destroyed', 'Buildings Built', 'Buildings Lost',
         'Buildings Destroyed', 'Supplies')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def sidecars():
    """Yield every crawler sidecar (match_<id>.json) under the GO archives."""
    for folder in FOLDERS:
        base = ROOT/'replays'/folder
        if not base.is_dir():
            continue
        for sub in os.scandir(base):
            if sub.is_dir():
                for entry in os.scandir(sub.path):
                    if entry.name.startswith('match_') and entry.name.endswith('.json'):
                        yield Path(entry.path)


def recorded():
    done = set()
    if OUT.exists():
        with OUT.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    done.add(json.loads(line)['match_id'])
    return done


def header_slots(side):
    return [(p['slot'], p['name']) for p in side.get('replay_header', {}).get('players', [])]


def assign_slots(players, slots):
    """Attach header slots by exact name; one unmatched pair is resolved by elimination."""
    names = {name: slot for slot, name in slots}
    for player in players:
        player['slot'] = names.get(player['name'])
    free = [slot for slot, _ in slots if slot not in {p['slot'] for p in players}]
    missing = [p for p in players if p['slot'] is None]
    if len(free) == 1 and len(missing) == 1:
        missing[0]['slot'] = free[0]
    return players


def from_listing(side):
    """Players and results from the saved listing row, or None when it shows no rating changes."""
    factions, text = side.get('listing_factions') or [], side.get('listing_text') or ''
    if len(factions) != 2:
        return None
    pattern = (re.escape(factions[0]) + r' (?P<n0>.+?) ' + DELTA.format(k=0) + r'.*? VS '
               + re.escape(factions[1]) + r' (?P<n1>.+?) ' + DELTA.format(k=1))
    match = re.search(pattern, text)
    if not match:
        return None
    players = []
    for k in (0, 1):
        delta = int(match['d%d' % k])
        players.append({'name': match['n%d' % k], 'faction': factions[k], 'delta': delta,
                        'board': match['b%d' % k], 'rating_before': int(match['a%d' % k]),
                        'rating_after': int(match['c%d' % k])})
    if not (players[0]['delta'] > 0 > players[1]['delta'] or players[1]['delta'] > 0 > players[0]['delta']):
        return None  # zero or same-sign changes: leave it to the page
    for player in players:
        player['result'] = 'won' if player['delta'] > 0 else 'lost'
    return assign_slots(players, header_slots(side))


class PageText(HTMLParser):
    """Visible text in order, with a '<player>' marker at each player profile link."""

    def __init__(self):
        super().__init__()
        self.tokens, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'svg'):
            self.skip += 1
        if tag == 'a' and '/zh/player/' in dict(attrs).get('href', ''):
            self.tokens.append(('<player>', dict(attrs)['href'].rsplit('/', 1)[1]))

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'svg'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.tokens.append(data.strip())


def from_page(markup):
    """Per-player rows of the match page: name, site player id, Result text, stats."""
    parser = PageText()
    parser.feed(markup)
    tokens = parser.tokens
    if 'Result' not in tokens:
        return []
    marks = [i for i, t in enumerate(tokens) if isinstance(t, tuple) and i > tokens.index('Result')]
    players = []
    for n, i in enumerate(marks):
        end = marks[n+1] if n+1 < len(marks) else len(tokens)
        row = [t for t in tokens[i+1:end] if isinstance(t, str)]
        if not row:
            continue
        try:
            cut = row.index('Units Created')
        except ValueError:
            continue
        head = row[:cut]  # name, army (shown twice), result, and on rated matches the rating change
        stats = {}
        for label in STATS:
            if label in row and row.index(label)+1 < len(row):
                value = row[row.index(label)+1].replace(',', '').replace('$', '')
                value = '0' if value == '—' else value  # the page shows an em dash for zero
                stats[label.lower().replace(' ', '_')] = int(value) if value.isdigit() else None
        words = [t for t in head[1:] if t in ('Won', 'Lost')]
        result = words[-1].lower() if words else (head[-1] if len(head) >= 3 else None)
        players.append({'name': head[0], 'site_player_id': tokens[i][1], 'result': result, 'stats': stats})
    return players


def fetch_page(client, side):
    import harvest_china_replays as harvest
    url = side['match_url']
    if not re.fullmatch(re.escape(harvest.BASE) + r'/zh/match/\d+', url):
        raise ValueError('Unexpected match URL: ' + url)
    return client.request(url).decode('utf-8', 'replace')


def client(delay):
    import harvest_china_replays as harvest
    return harvest.Client(delay=delay)


def keyed_sidecars(keys_file):
    """Sidecars of just the listed match keys (one 'pgo:<id>' per line), found via the cleaning metrics' paths."""
    import replay_cleaning
    wanted = {line.strip() for line in Path(keys_file).read_text(encoding='utf-8').splitlines() if line.strip()}
    found = {}
    with open(replay_cleaning.METRICS, encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('match_key') in wanted and row['match_key'] not in found:
                found[row['match_key']] = (ROOT/row['path']).with_suffix('.json')
    print('keys', len(wanted), 'sidecars found', len(found), flush=True)
    return [path for path in found.values() if path.exists()]


def backfill(args):
    done = recorded()
    counts = {'listing': 0, 'page': 0, 'no_result_listed': 0, 'failed': 0, 'already': 0}
    web = client(args.delay) if args.fetch else None
    fetched = 0
    with OUT.open('a', encoding='utf-8') as out:
        for path in (keyed_sidecars(args.keys) if args.keys else sidecars()):
            try:
                side = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                counts['failed'] += 1
                continue
            match_id = side.get('match_id')
            if match_id in done:
                counts['already'] += 1
                continue
            players, source = from_listing(side), 'listing'
            if players is None:
                if not args.fetch or (args.limit and fetched >= args.limit):
                    counts['no_result_listed'] += 1
                    continue
                try:
                    players, source = assign_slots(from_page(fetch_page(web, side)), header_slots(side)), 'page'
                except Exception as error:  # network or layout trouble: count it, retry on the next run
                    counts['failed'] += 1
                    print('failed', match_id, error, flush=True)
                    continue
                fetched += 1
            out.write(json.dumps({'match_key': 'pgo:%d' % match_id, 'match_id': match_id, 'source': source,
                                  'players': players, 'recorded_utc': now()}, ensure_ascii=False)+'\n')
            out.flush()
            done.add(match_id)
            counts[source] += 1
            if fetched and source == 'page' and fetched % 100 == 0:
                print('pages fetched', fetched, counts, flush=True)
    print(json.dumps(counts))


def verify(args):
    """Fetch pages for a random sample of listing-derived results and compare winners."""
    rows = [json.loads(line) for line in OUT.open(encoding='utf-8') if line.strip()]
    rows = [r for r in rows if r['source'] == 'listing']
    paths = {}
    for path in sidecars():
        paths[int(path.stem.split('_')[1])] = path
    random.seed(args.seed)
    web, agree, disagree = client(args.delay), 0, []
    for row in random.sample(rows, min(args.sample, len(rows))):
        side = json.loads(paths[row['match_id']].read_text(encoding='utf-8'))
        page = {p['name']: p['result'] for p in from_page(fetch_page(web, side))}
        listing = {p['name']: p['result'] for p in row['players']}
        if page == listing:
            agree += 1
        else:
            disagree.append({'match_id': row['match_id'], 'listing': listing, 'page': page})
    print(json.dumps({'checked': agree+len(disagree), 'agree': agree, 'disagree': disagree}, ensure_ascii=False))


def summary(_):
    rows = [json.loads(line) for line in OUT.open(encoding='utf-8') if line.strip()]
    by_source, unslotted, decided = {}, 0, 0
    for row in rows:
        by_source[row['source']] = by_source.get(row['source'], 0)+1
        unslotted += any(p.get('slot') is None for p in row['players'])
        decided += sorted(p['result'] for p in row['players']) == ['lost', 'won']
    print(json.dumps({'matches': len(rows), 'by_source': by_source, 'one_won_one_lost': decided,
                      'with_unslotted_player': unslotted}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('backfill')
    b.add_argument('--fetch', action='store_true', help='fetch match pages for matches without listed rating changes')
    b.add_argument('--limit', type=int, default=0, help='at most this many page fetches (0 = no limit)')
    b.add_argument('--delay', type=float, default=5, help='seconds between requests to the site')
    b.add_argument('--keys', help="only these matches: a file of 'pgo:<id>' lines (sidecars found via cleaning metrics)")
    v = sub.add_parser('verify')
    v.add_argument('--sample', type=int, default=30)
    v.add_argument('--seed', type=int, default=0)
    v.add_argument('--delay', type=float, default=5)
    sub.add_parser('summary')
    args = parser.parse_args()
    {'backfill': backfill, 'verify': verify, 'summary': summary}[args.command](args)


if __name__ == '__main__':
    main()
