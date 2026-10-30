"""Shared checkpoint IO and native gameplay evaluation; no run-owned code."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from engine_runtime import isolation_profile, clone_file, preflight
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'learner-runs/state'
CPU = GPU = ROOT/'.venv-torch-rocm/Scripts/python.exe'
FLAGS = subprocess.CREATE_NO_WINDOW

def _unconfigured(*args, **kwargs):
    raise RuntimeError('Configure shared gameplay helpers through run_prepared_training.py')

source_guard = status = wait_for_resources = run_phase = _unconfigured

def play_root():
    return Path(load(HERE/'plan.json').get('gameplay_work_root',ROOT/'learner-runs/gameplay-work'))


CASES = [
    dict(name='00-china-side-0', map='[RANK] TD NoBugsCars ZH v1', faction='FactionChina', side=0, seed=901900),
    dict(name='01-china-side-1', map='[RANK] TD NoBugsCars ZH v1', faction='FactionChina', side=1, seed=901900),
    dict(name='02-america-side-0', map='Tournament Desert', faction='FactionAmerica', side=0, seed=901901),
    dict(name='03-america-side-1', map='Tournament Desert', faction='FactionAmerica', side=1, seed=901901),
    dict(name='04-gla-side-0', map='[RANK] TD NoBugsCars ZH v1', faction='FactionGLA', side=0, seed=901902),
    dict(name='05-gla-side-1', map='[RANK] TD NoBugsCars ZH v1', faction='FactionGLA', side=1, seed=901902),
]

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.02)

def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default

def log(message):
    print(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **message}), flush=True)

def stop_owned_tree(pid):
    result = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30,
        creationflags=FLAGS)
    if result.returncode not in (0, 128):
        raise RuntimeError(f'Could not stop owned phase process {pid}: {result.stdout[-1000:]}')

def finite_tree(value):
    if isinstance(value, float):
        return value == value and abs(value) != float('inf')
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True

def setup_play_runtime():
    destination = play_root()
    runtime, profile = destination/'runtime', destination/'profile'
    if not runtime.exists():
        def ignore(directory, names):
            return [name for name in names if Path(name).suffix.lower() in ('.log', '.dmp', '.tmp', '.jsonl')]
        shutil.copytree(Path(load(HERE/'plan.json')['native']['exe']).parent, runtime, copy_function=clone_file, ignore=ignore)
    if not profile.exists():
        isolation_profile(ROOT/'replay-runtime/gameplay-profile', profile,
                          maps=['[RANK] TD NoBugsCars ZH v1', 'Tournament Desert'])
    expected = {
        '[RANK] TD NoBugsCars ZH v1': 'ed468fe5a5cb4e2145fa0b37bc5d2a1d4eec58b55fba58395deebe9f17dbd083',
        'Tournament Desert': 'db429b884bc593971feb9d545ada1f1d0aea114f2262b052fe46d81f9f034d01'}
    # Verify the actual preserved gameplay files against the runbook pins.
    for name in expected:
        path = profile/'Maps'/name
        assert path.is_dir() and not (path/'map.ini').exists()
        assert digest(path/(name+'.map')) == expected[name]
    native=load(HERE/'plan.json')['native']
    executable=runtime/Path(native.get('exe','current-60.exe')).name
    assert digest(executable) == native['sha256']
    os.environ['GENERALS_AI_USER_DATA'] = str(profile.resolve())
    preflight(executable, profile, destination/'profile-preflight.log', 120)
    return executable, profile

def score_game(path, summary, audit, catalogue):
    side = str(audit['controlled_players'][0])
    building_types = {name for key, names in catalogue['action_menus'].items()
                      if key.endswith('|build') for name in names}
    seen, completed, produced, built = set(), set(), Counter(), Counter()
    unloads = 0
    income_increases = 0
    first = {}
    with __import__('gzip').open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('record') != 'transition':
                continue
            before, after = row['observations'][side], row['next_observations'][side]
            if not seen:
                seen.update(entity['id'] for entity in before['own_entities'])
                completed.update(entity['id'] for entity in before['own_entities']
                                 if entity.get('construction_percent', -1) < 0)
            feedback = [item for item in row['action_feedback'] if str(item['player']) == side]
            if not any(item['accepted'] and item['action']['kind'] == 'sell' for item in feedback):
                income_increases += max(0, after['player']['cash']-before['player']['cash'])
            previous = {entity['id']: entity for entity in before['own_entities']}
            for entity in after['own_entities']:
                eid, kind = entity['id'], entity['type']
                if eid not in seen:
                    seen.add(eid)
                    if kind not in building_types and entity.get('producer_id'):
                        produced[kind] += 1
                        first.setdefault('produced_unit_seconds', row['next_frame']/60)
                if (kind in building_types and entity.get('construction_percent', -1) < 0
                        and eid not in completed):
                    completed.add(eid)
                    built[kind] += 1
                    first.setdefault('completed_building_seconds', row['next_frame']/60)
                old_cargo = previous.get(eid, {}).get('task', {}).get('cargo', {})
                new_cargo = entity.get('task', {}).get('cargo', {})
                if old_cargo.get('boxes', 0) > 0 and new_cargo.get('boxes') == 0:
                    unloads += 1
                    first.setdefault('cargo_unload_seconds', row['next_frame']/60)
    reward = summary['rewards'][side]
    outcome = ('time limit' if summary['truncated'] else
               ('win' if reward > 0 else 'loss' if reward < 0 else 'draw'))
    return {'outcome': outcome, 'seconds': summary['frame']/60, 'reward': reward,
        'completed_buildings': dict(built), 'produced_units': dict(produced),
        'observed_cargo_unloads': unloads, 'positive_cash_change_proxy': income_increases,
        'first_events': first, 'accepted': audit['accepted_actions'],
        'rejected': audit['rejected_actions'], 'final_owned_types': audit['final_owned_types']}

def play_all(plan):
    engine, profile = setup_play_runtime()
    checkpoint = Path(plan['checkpoint'])
    checkpoint_sha = plan['checkpoint_sha256']
    assert digest(checkpoint) == checkpoint_sha
    catalogue = load(Path(plan['action_catalog']))
    previous = {row['case']['name']: row for row in plan['gameplay']}
    completed = []
    for case in CASES:
        if case['name'] in previous and previous[case['name']].get('status') == 'completed':
            assert previous[case['name']]['checkpoint_sha256'] == checkpoint_sha
            completed.append(previous[case['name']])
            continue
        source_guard()
        existing = sorted((play_root()).glob(case['name']+'*'))
        recoverable = None
        for candidate in existing:
            summary_path, episode_path = candidate/'summary.json', candidate/'episode.jsonl'
            if not summary_path.is_file() or not episode_path.is_file():
                continue
            prior_summary = load(summary_path)
            if (prior_summary.get('checkpoint_sha256') == checkpoint_sha
                    and prior_summary.get('engine_sha256') == plan['native']['sha256']
                    and (prior_summary.get('terminated') or prior_summary.get('truncated'))):
                recoverable = candidate
                break
        recovered = recoverable is not None
        if recovered:
            output = recoverable
            suffix = '' if output.name == case['name'] else output.name[len(case['name']):]
            summary = load(output/'summary.json')
            log({'event': 'recovering_completed_game_after_resource_stop',
                 'case': case['name'], 'output': str(output)})
            game_phase = {'seconds': None, 'minimum_available_ram_bytes': None,
                          'resource_interruption': False, 'returncode': 0}
        else:
            index = 1
            while True:
                suffix = '' if index == 1 else f'-attempt-{index:03d}'
                output = play_root()/(case['name']+suffix)
                if not output.exists():
                    break
                index += 1
        map_path = profile/'Maps'/case['map']/(case['map']+'.map')
        command = [str(GPU), '-B', str(ROOT/'scripts/evaluate_events.py'),
            '--checkpoint', str(checkpoint), '--exe', str(engine), '--map', str(map_path),
            '--opponent-kind', 'hard_ai', '--bot-faction', case['faction'],
            '--learning-side', str(case['side']), '--seed', str(case['seed']),
            '--max-frames', '36000', '--ticks', '6', '--output', str(output)]
        if not recovered:
            game_phase = run_phase('play-'+case['name']+suffix, command, native=True)
            if game_phase['resource_interruption']:
                plan['gameplay_attempts'].append({'case': case, 'output': str(output),
                    'status': 'resource_interruption', 'seconds': game_phase['seconds']})
                save(HERE/'plan.json', plan)
                continue
            if game_phase['returncode']:
                raise RuntimeError(f'Native game failed. Inspect {game_phase["log"]}.')
            summary = load(output/'summary.json')
        assert summary['checkpoint_sha256'] == checkpoint_sha
        assert summary['engine_sha256'] == plan['native']['sha256']
        audit_log = HERE/'logs'/f'audit-{case["name"]}{suffix}.log'
        audit_phase = run_phase('audit-'+case['name']+suffix,
            [str(CPU), '-B', str(ROOT/'scripts/audit_self_play.py'), str(output/'episode.jsonl')])
        if audit_phase['returncode']:
            raise RuntimeError(f'Rollout audit failed. Inspect {audit_phase["log"]}.')
        audit = json.loads(Path(audit_phase['log']).read_text(encoding='utf-8'))
        save(output/'audit.json', audit)
        assert audit['steps'] > 0 and len(audit['controlled_players']) == 1
        assert audit['terminated'] or audit['truncated']
        payload = output/'episode.jsonl'
        raw_sha = digest(payload)
        packed = output/'episode.jsonl.gz'
        with payload.open('rb') as source, packed.open('wb') as compressed:
            with gzip.GzipFile(fileobj=compressed, mode='wb', compresslevel=6, mtime=0) as target:
                shutil.copyfileobj(source, target, 1024*1024)
        with gzip.open(packed, 'rb') as restored:
            restored_sha = hashlib.file_digest(restored, 'sha256').hexdigest()
        assert restored_sha == raw_sha
        payload.unlink()
        result = {'status': 'completed', 'case': case, 'checkpoint_sha256': checkpoint_sha,
            'engine_sha256': summary['engine_sha256'], 'map_sha256': summary['map_sha256'],
            'seed': case['seed'], 'elapsed_seconds': game_phase['seconds'],
            'recovered_after_resource_stop': recovered,
            'summary': summary, 'audit': audit,
            'score': score_game(packed, summary, audit, catalogue),
            'trajectory_gzip_sha256': digest(packed), 'trajectory_raw_sha256': raw_sha,
            'phase_resources': game_phase['minimum_available_ram_bytes']}
        plan['gameplay'].append(result)
        plan['gameplay_attempts'].append({'case': case, 'output': str(output),
            'status': 'completed_recovered' if recovered else 'completed',
            'elapsed_seconds': game_phase['seconds']})
        save(HERE/'plan.json', plan)
        completed.append(result)
        log({'event': 'game_completed', 'case': case['name'], 'score': result['score']})
    return completed
