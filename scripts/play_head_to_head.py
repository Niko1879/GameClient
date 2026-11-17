"""Head to head between two checkpoints: China mirror, both sides, both evaluation maps, sampled play.

Each game is evaluate_events.py --opponent-kind policy. Checkpoint A is the learning
side; side 1 reverses the actors, so A holds controlled_players[side]. Rerun to
continue: games with a finished summary.json are kept. Each worker slot gets its
own private profile. Refuses to start a game below 3 GiB available and stops all
games below 1 GiB.

  python -B scripts/play_head_to_head.py --a A.npz --b B.npz --output DIR
      [--a-rating 2000 --b-rating 1730 --rating-view mirror]
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import cycle_runtime as cycle
from engine_runtime import PROFILE_ENV, disable_power_throttling, isolation_profile, preflight
from pipeline_resources import tree
import run_prepared_training as runner
from openers import NAMED

MAPS = {'[RANK] TD NoBugsCars ZH v1': 'ed468fe5a5cb4e2145fa0b37bc5d2a1d4eec58b55fba58395deebe9f17dbd083',
        'Tournament Desert': 'db429b884bc593971feb9d545ada1f1d0aea114f2262b052fe46d81f9f034d01'}
START_RAM, STOP_RAM = 3*1024**3, 1024**3
FPS = 60


def games(seeds):
    out = []
    for m, name in enumerate(MAPS):
        for s in range(seeds):
            for side in (0, 1):
                out.append(dict(name=f'm{m}-s{s}-a{side}', map=name, side=side, seed=902000+100*m+s))
    return out


def side_stats(path, player):
    """Per-player stats from the trajectory: dozers, Command Center, income, production."""
    player = str(player)
    dozers, max_dozers, last_dozer, cc_gone, income, units, seen = set(), 0, None, None, 0, 0, set()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('record') != 'transition':
                continue
            before, after = row['observations'][player], row['next_observations'][player]
            sold = any(f['accepted'] and f['action']['kind'] == 'sell'
                       for f in row['action_feedback'] if str(f['player']) == player)
            if not sold:
                income += max(0, after['player']['cash']-before['player']['cash'])
            ents = after['own_entities']
            alive = [e['id'] for e in ents if e['type'].endswith('VehicleDozer')]
            dozers.update(alive)
            max_dozers = max(max_dozers, len(alive))
            minute = row['next_frame']/FPS/60
            if alive:
                last_dozer = None
            elif last_dozer is None:
                last_dozer = minute
            if cc_gone is None and not any(e['type'].endswith('CommandCenter') for e in ents):
                cc_gone = minute
            for e in ents:
                if e['id'] not in seen:
                    seen.add(e['id'])
                    if e.get('producer_id') and not e['type'].endswith(('Mine', 'ListeningOutpost')):
                        units += 1
    return dict(dozers_ever=len(dozers), max_dozers=max_dozers, no_dozer_from_min=last_dozer,
                no_cc_from_min=cc_gone, income_proxy=income, units_produced=units)


def finish(game, output, label):
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    controlled = summary.get('controlled_players') or json.loads(
        gzip.open(output/'episode.jsonl.gz', 'rt').readline())['controlled_players']
    a, b = controlled[game['side']], controlled[1-game['side']]
    ra = summary['rewards'][str(a)]
    outcome = 'time limit' if summary['truncated'] else ('A' if ra > 0 else 'B' if ra < 0 else 'draw')
    packed = output/'episode.jsonl.gz'
    return dict(game=game, outcome=outcome, minutes=summary['frame']/FPS/60,
                a_player=a, b_player=b, a=side_stats(packed, a), b=side_stats(packed, b),
                scheduling=summary.get('event_scheduling'), labels=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, default=5, help='Seeds per map; each seed is played from both sides')
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--minutes', type=float, default=20)
    parser.add_argument('--stagger', type=float, default=20,
                        help='Seconds between game starts: loading two checkpoints peaks well above a running game, '
                             'and simultaneous starts tripped the 1 GiB guard (12 Nov)')
    parser.add_argument('--a-rating', type=float, help='Rating input of A (default: the live play rating)')
    parser.add_argument('--b-rating', type=float, help='Rating input of B (default: the live play rating)')
    parser.add_argument('--rating-view', choices=('true', 'mirror'), default='true',
                        help="Opponent rating each side conditions on: the other's (true) or its own (mirror)")
    parser.add_argument('--a-opener', choices=('none', *sorted(NAMED)), default='none',
                        help='Strategy input z of A (scripts/openers.py NAMED)')
    parser.add_argument('--b-opener', choices=('none', *sorted(NAMED)), default='none', help='Strategy input z of B')
    parser.add_argument('--a-rating-input', action='store_true',
                        help='A was trained with the rating input (its run directory holds RATING_INPUT)')
    parser.add_argument('--b-rating-input', action='store_true', help='B was trained with the rating input')
    chinas = ('FactionChina', 'FactionChinaTankGeneral', 'FactionChinaNukeGeneral')
    parser.add_argument('--a-faction', choices=chinas, default='FactionChina', help='China general of A')
    parser.add_argument('--b-faction', choices=chinas, default='FactionChina', help='China general of B')
    parser.add_argument('--engine', help='Engine exe in replay-runtime/game (default: the pinned runner.ENGINE)')
    parser.add_argument('--record-replays', action='store_true',
                        help='Keep a viewable games/<name>/episode.rep per game (evaluate_events --record-replay)')
    args = parser.parse_args()
    disable_power_throttling()
    for key in ('a', 'b'):
        marked = (Path(getattr(args, key)).parent/'RATING_INPUT').exists()
        if marked != getattr(args, key+'_rating_input'):
            raise SystemExit(f'--{key}: RATING_INPUT marker {"present" if marked else "absent"}; '
                             f'pass --{key}-rating-input exactly when the model was trained with the rating input')
    engine_path = Path(args.engine).resolve() if args.engine else runner.ENGINE
    if engine_path.parent != runner.ENGINE.parent:
        raise SystemExit('--engine must sit in replay-runtime/game (the runtime copied per game)')
    output = Path(args.output).resolve()
    (output/'logs').mkdir(parents=True, exist_ok=True)
    frozen = {}
    for key in ('a', 'b'):
        frozen[key] = output/f'{key}.npz'
        if not frozen[key].exists():
            shutil.copy2(getattr(args, key), frozen[key])
    meta = dict(a=str(Path(args.a).resolve()), b=str(Path(args.b).resolve()),
                a_sha256=cycle.digest(frozen['a']), b_sha256=cycle.digest(frozen['b']),
                engine=str(engine_path), engine_sha256=cycle.digest(engine_path), minutes=args.minutes,
                **({'factions': dict(a=args.a_faction, b=args.b_faction)}
                   if (args.a_faction, args.b_faction) != ('FactionChina', 'FactionChina') else {}),
                ratings=dict(a=args.a_rating, b=args.b_rating, view=args.rating_view,
                             **({'a_input': True} if args.a_rating_input else {}),
                             **({'b_input': True} if args.b_rating_input else {})),
                **({'openers': dict(a=args.a_opener, b=args.b_opener)}
                   if (args.a_opener, args.b_opener) != ('none', 'none') else {}))
    previous = cycle.load(output/'meta.json')
    if previous and (any(previous.get(k) != meta.get(k) for k in ('a_sha256', 'b_sha256', 'engine_sha256', 'minutes', 'openers'))
                     or previous.get('ratings', dict(a=None, b=None, view='true')) != meta['ratings']):
        raise SystemExit('Output directory holds a different matchup; use a new --output')
    cycle.save(output/'meta.json', meta)

    runtime = output/'runtime'
    if not runtime.exists():
        def ignore(directory, names):
            return [n for n in names if Path(n).suffix.lower() in ('.log', '.dmp', '.tmp', '.jsonl')]
        shutil.copytree(runner.ENGINE.parent, runtime, copy_function=cycle.clone_file, ignore=ignore)
    engine = runtime/engine_path.name
    assert cycle.digest(engine) == meta['engine_sha256']
    profiles = []
    for slot in range(args.parallel):
        profile = output/f'profile-{slot}'
        if not (profile/'Maps').is_dir():
            isolation_profile(ROOT/'replay-runtime/gameplay-profile', profile, maps=list(MAPS))
        for name, sha in MAPS.items():
            assert cycle.digest(profile/'Maps'/name/(name+'.map')) == sha
        preflight(engine, profile, output/'logs'/f'preflight-{slot}.log', 120)
        profiles.append(profile)

    results_path = output/'results.json'
    results = {r['game']['name']: r for r in (cycle.load(results_path) or [])}
    todo = [g for g in games(args.seeds) if g['name'] not in results]
    failures = {}  # game name -> failed attempts; a failed game is retried later, not fatal for the others
    running = {}  # slot -> (game, process, family, log, start)
    frames = str(round(args.minutes*60*FPS))
    try:
        while todo or running:
            available = tree().get('available_ram_bytes', 0)
            if available < STOP_RAM:
                raise SystemExit('Under 1 GiB of RAM available; stopped. Rerun to continue.')
            for slot in range(args.parallel):
                if slot in running or not todo or tree().get('available_ram_bytes', 0) < START_RAM:
                    continue
                game = todo.pop(0)
                folder = output/'games'/game['name']
                if folder.exists():
                    shutil.rmtree(folder)  # an unfinished earlier attempt
                folder.parent.mkdir(exist_ok=True)
                profile = profiles[slot]
                # A game killed mid-recording (abort, crash) leaves its Last Replay behind, and
                # evaluate_events --record-replay then finds more than one (13 Nov): clear them first.
                for stale in (profile/'Replays').glob('00000000*.rep'):
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                map_path = profile/'Maps'/game['map']/(game['map']+'.map')
                command = [sys.executable, '-B', str(ROOT/'scripts/evaluate_events.py'),
                           '--checkpoint', str(frozen['a']), '--opponent', str(frozen['b']),
                           '--exe', str(engine), '--map', str(map_path), '--opponent-kind', 'policy',
                           '--learning-side', str(game['side']), '--seed', str(game['seed']),
                           '--max-frames', frames, '--ticks', '6', '--output', str(folder),
                           '--rating-view', args.rating_view,
                           '--faction', args.a_faction, '--opponent-policy-faction', args.b_faction,
                           '--opener', args.a_opener, '--opponent-opener', args.b_opener]
                if args.a_rating is not None:
                    command += ['--rating', str(args.a_rating)]
                if args.b_rating is not None:
                    command += ['--opponent-rating', str(args.b_rating)]
                if args.record_replays:
                    command += ['--record-replay']
                if args.a_rating_input:
                    command += ['--rating-input']
                if args.b_rating_input:
                    command += ['--opponent-rating-input']
                os.environ[PROFILE_ENV] = str(profile.resolve())
                log = (output/'logs'/(game['name']+'.log')).open('w', encoding='utf-8')
                process, family = runner.spawn_owned_python(command, log)
                running[slot] = (game, process, family, log, time.monotonic())
                print(json.dumps({'event': 'started', 'game': game['name'], 'slot': slot}), flush=True)
                time.sleep(args.stagger)  # the next start sees this one's load peak in available RAM
            for slot, (game, process, family, log, start) in list(running.items()):
                if process.poll() is None:
                    continue
                family.close()
                log.close()
                del running[slot]
                folder = output/'games'/game['name']
                if process.returncode:
                    failures[game['name']] = failures.get(game['name'], 0)+1
                    print(json.dumps({'event': 'game_failed', 'game': game['name'], 'attempt': failures[game['name']],
                                      'log': log.name}), flush=True)
                    if failures[game['name']] < 3:
                        todo.append(game)  # retried after the others
                    continue
                raw = folder/'episode.jsonl'
                with raw.open('rb') as source, gzip.open(folder/'episode.jsonl.gz', 'wb', compresslevel=6) as target:
                    shutil.copyfileobj(source, target, 1024*1024)
                raw.unlink()
                result = finish(game, folder, None)
                result['wall_seconds'] = time.monotonic()-start
                results[game['name']] = result
                cycle.save(results_path, sorted(results.values(), key=lambda r: r['game']['name']))
                print(json.dumps({'event': 'finished', 'game': game['name'], 'outcome': result['outcome'],
                                  'minutes': round(result['minutes'], 1)}), flush=True)
            time.sleep(3)
    finally:
        for game, process, family, log, start in running.values():
            family.close()
            if process.poll() is None:
                process.kill()
            log.close()
    tally = {}
    for r in results.values():
        tally[r['outcome']] = tally.get(r['outcome'], 0)+1
    print(json.dumps({'event': 'all_games_done', 'tally': tally, 'results': str(results_path)}), flush=True)


if __name__ == '__main__':
    main()
