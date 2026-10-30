"""Rebuild a viewable game replay (.rep) from a recorded Hard-AI/self-play rollout.

The recorded actions are resubmitted at their recorded frames, without the policy,
and every successor state must equal the recorded one. The resulting replay is then
checked by native headless playback (CRC comparisons, playback to EOF).
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess

from engine_runtime import PROFILE_ENV, clone_file, disable_power_throttling, isolation_profile
from generals_env import GeneralsEnv
from replay_training import require

ROOT = Path(__file__).resolve().parents[1]
WATCH = ROOT/'replay-watch'
# Concurrent engines suffix Last Replay with _InstanceNN.
LAST_REPLAY = '00000000*.rep'


def rows(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def watch_runtime(executable, maps):
    """Private runtime/profile: the user's own profile and Last Replay are never touched."""
    executable = Path(executable).resolve()
    runtime, profile = WATCH/'runtime', WATCH/'profile'
    if not runtime.exists():
        def ignore(directory, names):
            return [n for n in names if Path(n).suffix.lower() in ('.log', '.dmp', '.tmp', '.jsonl')
                    or (Path(n).suffix.lower() == '.exe' and n != executable.name)]
        shutil.copytree(executable.parent, runtime, copy_function=clone_file, ignore=ignore)
    isolation_profile(ROOT/'replay-runtime/gameplay-profile', profile, maps=maps)
    return runtime/executable.name, profile


def rebuild(episode, executable, name, max_frames):
    records = rows(episode)
    header = next(records)
    require(header['record'] == 'header' and header['source'] == 'self_play', 'Expected a self-play rollout')
    map_name = Path(header['map_path']).parent.name
    engine, profile = watch_runtime(executable, [map_name])
    os.environ[PROFILE_ENV] = str(profile.resolve())
    map_path = profile/'Maps'/map_name/(map_name+'.map')
    setup = header['setup']
    for stale in (profile/'Replays').glob(LAST_REPLAY):
        stale.unlink()
    with GeneralsEnv(engine, timeout=120, log_path=WATCH/'logs'/(name+'.engine.log')) as env:
        state = env.reset(map_path, seed=header['seed'], max_frames=max_frames,
                          starting_cash=setup['starting_cash'], opponent_kind=setup['opponent_kind'],
                          opponent_faction=setup['opponent_faction'], learning_side=setup['learning_side'],
                          record_replay=True)
        require(state['players'] == header['players'] and state['controlled_players'] == header['controlled_players'],
                'Rebuilt game has different players')
        steps = 0
        for record in records:
            if record['record'] == 'footer':
                footer = record
                break
            require(state['frame'] == record['frame'] and state['observations'] == record['observations'],
                    f'Rebuilt game diverged before frame {record["frame"]}')
            state = env.step({int(p): a for p, a in record['actions'].items()},
                             ticks=record['next_frame']-record['frame'], placement=header['execution_settings'])
            require(state['action_feedback'] == record['action_feedback'],
                    f'Action feedback diverged at frame {record["frame"]}')
            steps += 1
            transition = record
        require(steps == footer['steps'] and state['frame'] == footer['frame'] and env.done
                and state['rewards'] == footer['rewards'], 'Rebuilt game ended differently')
        require(state['observations'] == transition['next_observations'], 'Rebuilt final state differs')
    written = list((profile/'Replays').glob(LAST_REPLAY))
    require(len(written) == 1, 'Expected exactly one replay from the engine')
    last = written[0]
    target = profile/'Replays'/(name+'.rep')
    os.replace(last, target)
    validation = playback(engine, profile, target)
    return {'replay': str(target), 'steps': steps, 'frame': state['frame'], 'playback': validation}


def playback(engine, profile, replay):
    log = WATCH/'logs'/(replay.stem+'.playback.log')
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen([str(engine), '-headless', '-replay', replay.name], cwd=engine.parent,
                                   env={**os.environ, PROFILE_ENV: str(profile.resolve())},
                                   stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            disable_power_throttling(int(process._handle))
            process.wait(timeout=1800)
        except BaseException:
            process.kill()
            process.wait()
            raise
    result = process
    text = log.read_text(encoding='utf-8', errors='replace')
    summaries = [json.loads(line[len('ReplayValidation '):]) for line in text.splitlines()
                 if line.startswith('ReplayValidation ')]
    require(result.returncode == 0 and summaries, 'Native replay playback failed: '+str(log))
    summary = summaries[-1]
    require(summary['completed'] and not summary['crc_mismatch'] and summary['stream_eof']
            and summary['crc_comparisons'] > 0, 'Replay does not play back cleanly: '+json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', help='episode.jsonl or episode.jsonl.gz from a gameplay case')
    parser.add_argument('--exe', required=True, help='replay-capable engine build')
    parser.add_argument('--name', required=True, help='replay file name, without .rep')
    parser.add_argument('--max-frames', type=int, default=36000)
    args = parser.parse_args()
    (WATCH/'logs').mkdir(parents=True, exist_ok=True)
    print(json.dumps(rebuild(args.episode, args.exe, args.name, args.max_frames), indent=2))
