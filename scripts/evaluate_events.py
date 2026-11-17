"""Play one recorded episode: an event-policy checkpoint against Hard AI or another checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from engine_runtime import PROFILE_ENV
from event_policy import EventActor, ScienceBuyer
from generals_env import GeneralsEnv, run_episode
from generals_policy import PLAY_RATING, NeuralPolicy
from openers import NAMED, named_templates
from prepared_io import _digest, _write_json
from replay_bootstrap import check_live_provenance


def rating_view(own, other, view):
    """(own, opponent) rating inputs for one actor: 'true' sees the other side's rating,
    'mirror' assumes an evenly matched game at its own rating (isolates own-rating effects)."""
    return (own, other) if view == 'true' else (own, own)


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    policy, _, checkpoint = NeuralPolicy.load(args.checkpoint)
    policy.rating_input = args.rating_input
    with GeneralsEnv(args.exe, log_path=output/'engine.log') as env:
        engine_sha256 = _digest(args.exe)
        check_live_provenance(env.metadata, engine_sha256, checkpoint['metadata']['deployment'])
        # Tank/Nuke sides buy no sciences: science-orders.json holds vanilla China's trees (user, 15 Nov 2026).
        def science(faction):
            if faction == 'FactionChina':
                return None
            buyer = ScienceBuyer()
            buyer.orders, buyer.default = {}, []
            return buyer
        actors = [EventActor(policy, args.seed, deterministic=args.deterministic, science=science(args.faction))]
        actors[0].configure_ratings(*rating_view(args.rating, args.opponent_rating, args.rating_view))
        actors[0].configure_opener(named_templates(policy.config, args.opener))
        if args.opponent_kind == 'policy':
            other, _, _ = NeuralPolicy.load(args.opponent or args.checkpoint)
            other.rating_input = args.opponent_rating_input if args.opponent else args.rating_input
            actors.append(EventActor(other, args.seed+1, deterministic=args.deterministic,
                                     science=science(args.opponent_policy_faction)))
            actors[1].configure_ratings(*rating_view(args.opponent_rating, args.rating, args.rating_view))
            actors[1].configure_opener(named_templates(other.config, args.opponent_opener))
            if args.learning_side == 1:
                actors.reverse()
        policy_factions = None
        if (args.faction, args.opponent_policy_faction) != ('FactionChina', 'FactionChina'):
            if args.opponent_kind != 'policy':
                raise ValueError('--faction/--opponent-policy-faction apply to policy games')
            policy_factions = [None, None]
            policy_factions[args.learning_side] = args.faction
            policy_factions[1-args.learning_side] = args.opponent_policy_faction
        summary = run_episode(env, actors, args.map, seed=args.seed, max_frames=args.max_frames,
                              ticks=args.ticks, trajectory=output/'episode.jsonl', opponent_kind=args.opponent_kind,
                              opponent_faction=args.bot_faction, learning_side=args.learning_side,
                              record_replay=args.record_replay, policy_factions=policy_factions)
    if args.record_replay:
        # The engine wrote the private profile's Last Replay; keep it with the episode.
        # Concurrent engines suffix it with _InstanceNN; a private profile has only this one.
        written = list((Path(os.environ[PROFILE_ENV])/'Replays').glob('00000000*.rep'))
        if len(written) != 1:
            raise RuntimeError('Expected exactly one replay in the private profile')
        os.replace(written[0], output/'episode.rep')
    summary.update(checkpoint_sha256=_digest(args.checkpoint), engine_sha256=engine_sha256,
                   map_sha256=_digest(args.map), seed=args.seed, ticks=args.ticks, deterministic=args.deterministic,
                   factions=dict(checkpoint=args.faction, opponent=args.opponent_policy_faction),
                   ratings=dict(checkpoint=args.rating, opponent=args.opponent_rating, view=args.rating_view,
                                checkpoint_input=args.rating_input, opponent_input=args.opponent_rating_input),
                   openers=dict(checkpoint=args.opener, opponent=args.opponent_opener))
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
    chinas = ('FactionChina', 'FactionChinaTankGeneral', 'FactionChinaNukeGeneral')
    parser.add_argument('--faction', choices=chinas, default='FactionChina',
                        help='China general of the --checkpoint side in policy games (Tank/Nuke buy no sciences)')
    parser.add_argument('--opponent-policy-faction', choices=chinas, default='FactionChina',
                        help='China general of the --opponent side in policy games')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-frames', type=int, default=36000)
    parser.add_argument('--ticks', type=int, default=6)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--rating', type=float, default=PLAY_RATING, help='Rating input of the --checkpoint side')
    parser.add_argument('--opponent-rating', type=float, default=PLAY_RATING, help='Rating input of the --opponent side')
    parser.add_argument('--opener', choices=('none', *sorted(NAMED)), default='none',
                        help='Strategy input z of the --checkpoint side (scripts/openers.py NAMED)')
    parser.add_argument('--opponent-opener', choices=('none', *sorted(NAMED)), default='none',
                        help='Strategy input z of the --opponent side')
    parser.add_argument('--rating-input', action='store_true',
                        help='The --checkpoint model was trained with the rating input (GENERALS_RATING_INPUT=1)')
    parser.add_argument('--opponent-rating-input', action='store_true',
                        help='The --opponent model was trained with the rating input')
    parser.add_argument('--rating-view', choices=('true', 'mirror'), default='true',
                        help="Opponent rating each actor conditions on: the other side's (true) or its own (mirror)")
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
