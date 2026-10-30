"""Play one recorded episode: an event-policy checkpoint against Hard AI or another checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from engine_runtime import PROFILE_ENV
from event_policy import EventActor
from generals_env import GeneralsEnv, run_episode
from generals_policy import NeuralPolicy
from prepared_io import _digest, _write_json
from replay_bootstrap import check_live_provenance


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    policy, _, checkpoint = NeuralPolicy.load(args.checkpoint)
    with GeneralsEnv(args.exe, log_path=output/'engine.log') as env:
        engine_sha256 = _digest(args.exe)
        check_live_provenance(env.metadata, engine_sha256, checkpoint['metadata']['deployment'])
        actors = [EventActor(policy, args.seed, deterministic=args.deterministic)]
        if args.opponent_kind == 'policy':
            other, _, _ = NeuralPolicy.load(args.opponent or args.checkpoint)
            actors.append(EventActor(other, args.seed+1, deterministic=args.deterministic))
            if args.learning_side == 1:
                actors.reverse()
        summary = run_episode(env, actors, args.map, seed=args.seed, max_frames=args.max_frames,
                              ticks=args.ticks, trajectory=output/'episode.jsonl', opponent_kind=args.opponent_kind,
                              opponent_faction=args.bot_faction, learning_side=args.learning_side,
                              record_replay=args.record_replay)
    if args.record_replay:
        # The engine wrote the private profile's Last Replay; keep it with the episode.
        # Concurrent engines suffix it with _InstanceNN; a private profile has only this one.
        written = list((Path(os.environ[PROFILE_ENV])/'Replays').glob('00000000*.rep'))
        if len(written) != 1:
            raise RuntimeError('Expected exactly one replay in the private profile')
        os.replace(written[0], output/'episode.rep')
    summary.update(checkpoint_sha256=_digest(args.checkpoint), engine_sha256=engine_sha256,
                   map_sha256=_digest(args.map), seed=args.seed, ticks=args.ticks, deterministic=args.deterministic)
    _write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--opponent-kind', choices=('policy', 'hard_ai'), default='hard_ai')
    parser.add_argument('--opponent', help='Opponent checkpoint for policy games; defaults to the same checkpoint')
    parser.add_argument('--bot-faction', default='random', help='random or an engine faction name')
    parser.add_argument('--learning-side', type=int, choices=(0, 1), default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-frames', type=int, default=36000)
    parser.add_argument('--ticks', type=int, default=6)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--record-replay', action='store_true',
                        help=f'Also save a viewable episode.rep; requires a private profile in {PROFILE_ENV}')
    args = parser.parse_args()
    if args.record_replay and not os.environ.get(PROFILE_ENV):
        parser.error(f'--record-replay needs {PROFILE_ENV} so the user profile Last Replay is untouched')
    if not 1 <= args.ticks <= 600 or args.max_frames < 1:
        parser.error('ticks must be 1..600 and max-frames positive')
    evaluate(args)


if __name__ == '__main__':
    main()
