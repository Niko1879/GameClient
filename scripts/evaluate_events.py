"""Play an event-policy checkpoint against Hard AI, a human, or another checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from shutil import copy2

from engine_runtime import PROFILE_ENV, isolation_profile
from event_policy import LOOK_TICKS, science_buyer
from generals_env import GeneralsEnv, run_episode
from generals_policy import PLAY_APM, PLAY_RATING
from openers import NAMED, named_templates
from prepared_io import _digest, _write_json
from replay_bootstrap import check_live_provenance, check_live_rules


def rating_view(own, other, view):
    """(own, opponent) rating inputs for one actor: 'true' sees the other side's rating,
    'mirror' assumes an evenly matched game at its own rating (isolates own-rating effects)."""
    return (own, other) if view == 'true' else (own, own)


def configure_timing(actor, apm, look_ms):
    """A new policy's APM input and look interval. A memory-free checkpoint has neither: --apm fixes its pace."""
    if hasattr(actor, 'configure_pace'):
        actor.configure_pace(apm)
    else:
        actor.configure_apm(PLAY_APM if apm is None else apm)
        actor.configure_look(look_ms)


def evaluate(args, *, profile_source=None, write_trajectory=True):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if profile_source is not None:
        profile = isolation_profile(profile_source, output/'profile')
        source_map = Path(args.map)
        if source_map.is_relative_to(profile_source/'Maps'):
            args.map = str(profile/source_map.relative_to(profile_source))
        else:
            destination = profile/'Maps'/source_map.stem
            destination.mkdir(parents=True, exist_ok=True)
            for source in source_map.parent.glob(source_map.stem+'.*'):
                copy2(source, destination/source.name)
            args.map = str(destination/source_map.name)
        os.environ[PROFILE_ENV] = str(profile.resolve())
    # AlphaGenerals or, until its adoption, a memory-free checkpoint through the _old modules.
    from policy_checkpoints import actor_class, live_policy, load_policy
    policy, _, checkpoint = load_policy(args.checkpoint)
    policy.rating_input = args.rating_input
    device = getattr(args, 'device', None) or ('cuda' if args.headed else None)

    def live(reference):
        # Without a device, headless games use the NumPy reference policy (the comparison network has none).
        if device is None and not reference.config.get('network'):
            return reference
        return live_policy(reference, device=device or 'cpu')

    policy = live(policy)
    with GeneralsEnv(args.exe, log_path=output/'engine.log', headed=args.headed) as env:
        engine_sha256 = _digest(args.exe)
        deployment = checkpoint['metadata']['deployment']
        if args.headed:
            # The explicitly chosen local engine includes the headed bridge.
            check_live_rules(env.metadata, deployment)
        else:
            check_live_provenance(env.metadata, engine_sha256, deployment)
        science = science_buyer
        policy.constant_pace = args.constant_pace
        actors = [actor_class(policy)(policy, args.seed, deterministic=args.deterministic, science=science(args.faction))]
        actors[0].configure_ratings(*rating_view(args.rating, args.opponent_rating, args.rating_view))
        actors[0].configure_opener(named_templates(policy.config, args.opener))
        configure_timing(actors[0], args.apm, args.look_ms)
        if args.opponent_kind == 'policy':
            other, _, _ = load_policy(args.opponent or args.checkpoint)
            other.rating_input = args.opponent_rating_input if args.opponent else args.rating_input
            other = live(other)
            other.constant_pace = args.opponent_constant_pace
            actors.append(actor_class(other)(other, args.seed+1, deterministic=args.deterministic,
                                             science=science(args.opponent_policy_faction)))
            actors[1].configure_ratings(*rating_view(args.opponent_rating, args.rating, args.rating_view))
            actors[1].configure_opener(named_templates(other.config, args.opponent_opener))
            configure_timing(actors[1], args.opponent_apm, args.look_ms)
            if args.learning_side == 1:
                actors.reverse()
        policy_factions = None
        if (args.faction, args.opponent_policy_faction) != ('FactionChina', 'FactionChina'):
            policy_factions = ['FactionChina', 'FactionChina']
            policy_factions[args.learning_side] = args.faction
            policy_factions[1-args.learning_side] = args.opponent_policy_faction
        summary = run_episode(env, actors, args.map, seed=args.seed, max_frames=args.max_frames,
                              ticks=args.ticks, trajectory=output/'episode.jsonl' if write_trajectory else None,
                              opponent_kind=args.opponent_kind,
                              opponent_faction=args.bot_faction, learning_side=args.learning_side,
                              record_replay=args.record_replay, policy_factions=policy_factions)
        if args.headed and args.keep_window and not summary['truncated']:
            print('Match finished. Close the game window to return to the launcher.', flush=True)
            env.process.wait()
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
                   openers=dict(checkpoint=args.opener, opponent=args.opponent_opener),
                   apm=dict(checkpoint=args.apm, opponent=args.opponent_apm), look_ms=args.look_ms,
                   constant_pace=dict(checkpoint=args.constant_pace, opponent=args.opponent_constant_pace))
    _write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = make_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    evaluate(args)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--headed', action='store_true', help='Run windowed at normal speed with GPU inference')
    parser.add_argument('--device', help='Torch inference device (cuda = the ROCm/CUDA GPU, cpu); default: headless '
                        'games use the NumPy policy, headed games cuda')
    parser.add_argument('--keep-window', action='store_true', help='Keep a headed game open after victory or defeat')
    parser.add_argument('--opponent-kind', choices=('policy', 'hard_ai', 'human'), default='hard_ai')
    parser.add_argument('--opponent', help='Opponent checkpoint for policy games; defaults to the same checkpoint')
    parser.add_argument('--bot-faction', default='random', help='random or an engine faction name')
    parser.add_argument('--learning-side', type=int, choices=(0, 1), default=0)
    chinas = ('FactionChina', 'FactionChinaTankGeneral', 'FactionChinaNukeGeneral')
    parser.add_argument('--faction', choices=chinas, default='FactionChina',
                        help='China general of the checkpoint side (Tank/Nuke buy no sciences)')
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
    parser.add_argument('--apm', type=float, help='APM input of the --checkpoint side (default %d). A memory-free checkpoint '
                        'has no such input: there it fixes the commands per game minute, waits masked' % PLAY_APM)
    parser.add_argument('--opponent-apm', type=float, help='The same for the --opponent side')
    parser.add_argument('--constant-pace', action='store_true', help='The --checkpoint side acts at its APM input rate '
                        'in every state, never choosing no-op (pace off; a forced rate for other policies)')
    parser.add_argument('--opponent-constant-pace', action='store_true', help='The same for the --opponent side')
    parser.add_argument('--look-ms', type=float, default=1000*LOOK_TICKS/60, help='Game time between looks of both sides')
    parser.add_argument('--rating-input', action='store_true',
                        help='The --checkpoint model was trained with the rating input (GENERALS_RATING_INPUT=1)')
    parser.add_argument('--opponent-rating-input', action='store_true',
                        help='The --opponent model was trained with the rating input')
    parser.add_argument('--rating-view', choices=('true', 'mirror'), default='true',
                        help="Opponent rating each actor conditions on: the other side's (true) or its own (mirror)")
    parser.add_argument('--record-replay', action='store_true',
                        help=f'Also save a viewable episode.rep; requires a private profile in {PROFILE_ENV}')
    return parser


def validate_args(parser, args):
    if args.record_replay and not os.environ.get(PROFILE_ENV):
        parser.error(f'--record-replay needs {PROFILE_ENV} so the user profile Last Replay is untouched')
    if not 1 <= args.ticks <= 600 or args.max_frames < 1:
        parser.error('ticks must be 1..600 and max-frames positive')
    if (args.headed and args.opponent_kind == 'policy') or (not args.headed and args.opponent_kind == 'human'):
        parser.error('Headed games support human or Hard AI opponents. Headless games support policy or Hard AI.')
    if args.opponent_policy_faction != 'FactionChina' and args.opponent_kind != 'policy':
        parser.error('--opponent-policy-faction requires --opponent-kind policy')


if __name__ == '__main__':
    main()
