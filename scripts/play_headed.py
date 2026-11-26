"""Launch a windowed game with the locally trained Zero Hour policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

from engine_runtime import PROFILE_ENV, disable_power_throttling, isolation_profile
from evaluate_events import evaluate, make_parser
from event_policy import DECISION_LOG_ENV, LOOK_TICKS, science_buyer
from event_schedule import step_ticks
from generals_env import EngineClosed, placement_options
from generals_policy import OBSERVATION_VERSION
from prepared_io import _digest
from replay_bootstrap import check_live_rules

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'.cache'
GAME = ROOT/'replay-runtime/game'
PROFILE = ROOT/'replay-runtime/gameplay-profile'
CHECKPOINT = CACHE/'cloud-runs/v11-1500/best.npz'
BUILD = ROOT/'build/win32-vcpkg/GeneralsMD/Release/GeneralsOnlineZH.exe'
DEFAULT_MAP = PROFILE/'Maps/Tournament Desert/Tournament Desert.map'


def configure_gpu_cache():
    os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')
    for variable, name in (('MIOPEN_USER_DB_PATH', 'user-db'), ('MIOPEN_CUSTOM_CACHE_DIR', 'kernels')):
        path = CACHE/'miopen'/name
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(path))


def staged_engine(build):
    digest = _digest(build)
    target = GAME/f'engine-headed-{digest[:16]}.exe'
    if not target.exists():
        # The replay keeps using this build even after the next local compilation.
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(dir=GAME, suffix='.tmp', delete=False) as temp:
            temporary = Path(temp.name)
        try:
            shutil.copyfile(build, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def skirmish_worker(checkpoint, rating, device, apm, look_ms):
    """The engine owns the process and sends one observation at a time."""
    from evaluate_events import configure_timing
    from policy_checkpoints import actor_class, live_policy, load_policy

    def receive(line):
        if not line.startswith('GAI '):
            raise ValueError('Expected a GAI message from the engine')
        return json.loads(line[4:])

    request_id = 1

    def send(command, **payload):
        nonlocal request_id
        print(json.dumps(dict(request_id=request_id, command=command, **payload), allow_nan=False), flush=True)
        request_id += 1

    ready = receive(sys.stdin.readline())
    if ready['event'] != 'ready' or ready['protocol_version'] != 1:
        raise ValueError('Unsupported Neural AI handshake')
    policy, _, document = load_policy(checkpoint)
    policy.rating_input = (checkpoint.parent/'RATING_INPUT').is_file()
    deployment = document['metadata']['deployment']
    check_live_rules(ready, deployment)
    if ready['observation_policy']['version'] != OBSERVATION_VERSION or policy.config['placement_mode'] != 'fast':
        raise ValueError('Neural AI requires the current observation contract and fast build placement')
    actor = actor_class(policy)(live_policy(policy, device=device))
    configure_timing(actor, apm, look_ms)
    actor.configure_clock(ready['ticks_per_second'])
    actor.configure_ratings(rating, rating)
    placement = placement_options([actor])
    send('start')
    terrain = None
    # The first two minutes of what the model saw, one file per game (the newest 5 kept),
    # so an odd opening can be compared against headless games afterwards.
    traces = Path(os.environ[PROFILE_ENV])/'neural-ai-traces'
    traces.mkdir(exist_ok=True)
    for old in sorted(traces.glob('*.jsonl'))[:-4]:
        old.unlink(missing_ok=True)
    trace = open(traces/(datetime.now().strftime('%Y%m%d-%H%M%S')+'.jsonl'), 'w', encoding='utf-8')
    trace.write(json.dumps(dict(ready=ready))+'\n')
    for line in sys.stdin:
        state = receive(line)
        if state['event'] == 'error':
            raise RuntimeError(state['message'])
        if state['terminated'] or state['truncated']:
            return
        player, = state['controlled_players']
        if terrain is None:
            terrain = state['map']
            # The lobby's China general for this slot (China, Tank or Nuke General).
            actor.science = science_buyer(state['observations'][str(player)]['player']['faction'])
            actor.configure_enemy(state['setup']['opponent_faction'])
            actor.policy.warm_up(actor, state['observations'][str(player)], terrain, state['frame'])
        actor.feedback(state['action_feedback'])
        actions = actor.act(state['observations'][str(player)], terrain, state['frame'])
        ticks = step_ticks([actor], state['frame'], 6, ready['ticks_per_second'])
        send('step', ticks=ticks, actions={str(player): actions}, **placement)
        if state['frame'] <= 7200:
            trace.write(json.dumps(dict(state=state, actions=actions))+'\n')
            trace.flush()


def decision_console():
    """A console window that follows the model's per-look decisions (event_policy.decision_line)."""
    log = CACHE/'headed-play'/'decisions.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('', encoding='utf-8')
    os.environ[DECISION_LOG_ENV] = str(log)
    script = ("$host.UI.RawUI.WindowTitle = 'Model decisions'; "
              f"Get-Content -LiteralPath '{log}' -Wait -Encoding UTF8")
    console = subprocess.Popen(['powershell.exe', '-NoProfile', '-NoExit', '-Command', script],
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
    disable_power_throttling(int(console._handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--watch', action='store_true', help='Observe the model against Hard AI')
    modes.add_argument('--skirmish', action='store_true', help='Open the game with Neural AI available in Skirmish')
    modes.add_argument('--replay', type=Path, metavar='SESSION', help='Open a recorded session with its game data')
    modes.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--build', type=Path, default=BUILD, help='Locally built engine executable')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--map', type=Path, default=DEFAULT_MAP)
    parser.add_argument('--faction', default='FactionChina', help='Human or Hard AI faction. Use an engine name or random.')
    parser.add_argument('--model-faction', default='FactionChina', help='Model faction in a direct match')
    parser.add_argument('--rating', type=float, default=1900)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-frames', type=int, default=3600000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu', help='Inference device: cpu, or cuda for the ROCm/CUDA build')
    parser.add_argument('--apm', type=float, help='Playing pace (evaluate_events --apm)')
    parser.add_argument('--look-ms', type=float, default=1000*LOOK_TICKS/60, help='Game time between the model\'s looks')
    parser.add_argument('--trajectory', action='store_true', help='Save the full observation and action log')
    parser.add_argument('--decisions', action='store_true',
                        help='Open a console that prints each look: game time, P(act), no-op or the order, '
                             'and the five likeliest orders as sampled (current pipeline)')
    args = parser.parse_args()
    if args.decisions and not (args.worker or args.replay):
        decision_console()
    if args.device != 'cpu':
        configure_gpu_cache()
    if args.worker:
        skirmish_worker(args.checkpoint.resolve(), args.rating, args.device, args.apm, args.look_ms)
        return
    if args.replay:
        session = args.replay.resolve()
        summary = json.loads((session/'summary.json').read_text(encoding='utf-8'))
        engine = GAME/f"engine-headed-{summary['engine_sha256'][:16]}.exe"
        if not engine.is_file() or _digest(engine) != summary['engine_sha256']:
            parser.error(f'The replay needs its recording engine at {engine}')
        profile = session/'profile'
        shutil.copy2(session/'episode.rep', profile/'Replays/episode.rep')
    else:
        if not args.build.is_file() or not args.checkpoint.is_file():
            parser.error(f'Expected a local engine at {args.build} and a checkpoint at {args.checkpoint}')
        engine = staged_engine(args.build.resolve())
        if args.skirmish:
            profile = isolation_profile(PROFILE, CACHE/'headed-play/skirmish-profile')
            os.environ['GENERALS_NEURAL_ACTOR'] = subprocess.list2cmdline([
                sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                '--checkpoint', str(args.checkpoint.resolve()), '--rating', str(args.rating), '--device', args.device,
                '--look-ms', str(args.look_ms), *([] if args.apm is None else ['--apm', str(args.apm)])])
        else:
            if not args.map.is_file() or not 1 <= args.max_frames <= 3600000:
                parser.error('A map file and a frame limit in 1..3600000 are required')
            output = args.output or CACHE/'headed-play'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            match = make_parser().parse_args([
                '--headed', '--keep-window', '--record-replay', '--exe', str(engine), '--map', str(args.map.resolve()),
                '--checkpoint', str(args.checkpoint.resolve()), '--output', str(output.resolve()), '--seed', str(args.seed),
                '--max-frames', str(args.max_frames), '--rating', str(args.rating), '--opponent-rating', str(args.rating),
                '--faction', args.model_faction, '--bot-faction', args.faction,
                '--opponent-kind', 'hard_ai' if args.watch else 'human'])
            match.device, match.apm, match.look_ms = args.device, args.apm, args.look_ms
            match.rating_input = (args.checkpoint.resolve().parent/'RATING_INPUT').is_file()
            try:
                evaluate(match, profile_source=PROFILE, write_trajectory=args.trajectory)
            except EngineClosed:
                # The window can also close before an initial state is available.
                print(f'Game closed. Session files are in {output.resolve()}', flush=True)
            return
    subprocess.run([str(engine), '-win', '-quickstart', '-noshellmap'],
                   cwd=GAME, env={**os.environ, PROFILE_ENV: str(profile)}, check=True)


if __name__ == '__main__':
    main()
