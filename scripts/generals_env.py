"""Synchronous, two-player headless Generals environment; standard library only."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading


class EnvironmentError(RuntimeError):
    pass


class GeneralsEnv:
    """One engine process per environment. Policies maintain separate memory per player.

    reset() and step() return a joint state; each policy is handed only
    state['observations'][str(player)]. No game time passes between requests.
    """
    def __init__(self, executable, *, timeout=60, log_path=None):
        self.executable = Path(executable).resolve()
        self.timeout = timeout
        self.players = []
        self.controlled_players = []
        self.done = True
        self.request_id = 0
        self.responses = queue.Queue()
        self.log = Path(log_path).open('w', encoding='utf-8') if log_path else None
        self.process = subprocess.Popen(
            [str(self.executable), '-headless', '-selfPlay', '-noshellmap'],
            cwd=self.executable.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            if os.name == 'nt':
                # Repository helper, not standard library: imported only where it applies.
                from engine_runtime import disable_power_throttling
                disable_power_throttling(int(self.process._handle))
            self.metadata = self._response(0)
            if self.metadata.get('event') != 'ready' or self.metadata.get('protocol_version') != 1:
                raise EnvironmentError('Unsupported environment handshake')
        except BaseException:
            self.close(force=True)
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith('GAI '):
                    try:
                        self.responses.put(json.loads(line[4:]))
                    except ValueError as error:
                        self.responses.put(error)
                elif self.log:
                    self.log.write(line)
                    self.log.flush()
        finally:
            self.responses.put(EnvironmentError('Engine output closed'))

    def _response(self, request_id):
        try:
            response = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close(force=True)
            raise EnvironmentError(f'Engine response timed out after {self.timeout}s') from None
        if isinstance(response, BaseException):
            raise EnvironmentError(str(response))
        if response.get('request_id') != request_id:
            raise EnvironmentError('Out-of-order engine response')
        if response.get('event') == 'error':
            raise EnvironmentError(response.get('message', 'Engine request failed'))
        return response

    def _request(self, command, **payload):
        self.request_id += 1
        request = {'request_id': self.request_id, 'command': command, **payload}
        try:
            self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise EnvironmentError('Engine process is unavailable') from None
        return self._response(self.request_id)

    def reset(self, map_path, *, seed=1, max_frames=36000, starting_cash=10000,
              opponent_kind='policy', opponent_faction='random', learning_side=0, record_replay=False):
        """record_replay writes the episode as the private profile's Last Replay (00000000.rep)."""
        map_path = Path(map_path).resolve()
        if not map_path.is_file():
            raise ValueError(f'Map does not exist: {map_path}')
        if opponent_kind not in ('policy', 'hard_ai') or learning_side not in (0, 1):
            raise ValueError('Invalid opponent kind or learning side')
        result = self._request('reset', map=str(map_path), seed=seed,
                               max_frames=max_frames, starting_cash=starting_cash,
                               opponent_kind=opponent_kind, opponent_faction=opponent_faction,
                               learning_side=learning_side, record_replay=record_replay)
        self.players = result['players']
        self.controlled_players = result['controlled_players']
        self.done = result['terminated'] or result['truncated']
        return result

    def step(self, actions=None, *, ticks=6, placement=None):
        """Advance; placement carries the policy's construction execution settings."""
        if self.done:
            raise EnvironmentError('Episode finished; reset first')
        if not isinstance(ticks, int) or not 1 <= ticks <= 600:
            raise ValueError('ticks must be 1..600')
        actions = {str(p): batch for p, batch in (actions or {}).items()}
        if not set(actions).issubset(map(str, self.controlled_players)):
            raise ValueError('Unknown controlled player')
        result = self._request('step', actions=actions, ticks=ticks, **(placement or {}))
        self.done = result['terminated'] or result['truncated']
        return result

    def placement(self, player, builder, template, angle_index, bins):
        if self.done or player not in self.controlled_players:
            raise EnvironmentError('Placement query requires a live controlled player')
        return self._request('placement', player=player, builder=builder, template=template,
                             angle_index=angle_index, bins=bins)['placement']

    def close(self, force=False):
        if getattr(self, 'process', None) is None:
            return
        try:
            if self.process.poll() is None:
                if force:
                    self.process.kill()
                else:
                    try:
                        self._request('close')
                    except (EnvironmentError, OSError):
                        self.process.kill()
                self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.reader.join(timeout=2)
            if self.log:
                self.log.close()
                self.log = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def placement_options(policies):
    """Fast mode snaps each construction order to the nearest legal site in a bounded disc.

    Native placement settings apply to the whole step, so every actor in one
    session must share them.
    """
    settings = {(actor.policy.config['build_snap_radius'], actor.policy.config['build_snap_step']) for actor in policies}
    if len(settings) != 1:
        raise ValueError('Policies require different placement execution; evaluate in separate episodes')
    radius, step = settings.pop()
    return dict(build_snap_radius=radius, build_snap_step=step) if radius else {}


def run_episode(env, policies, map_path, *, seed=1, max_frames=36000, ticks=6, trajectory=None,
                opponent_kind='policy', opponent_faction='random', learning_side=0, record_replay=False):
    """Each policy implements reset(), act(observation, map_features, frame) and feedback().

    Reward is terminal win/loss only; timeout is truncation with zero reward.
    """
    from event_schedule import step_ticks
    from generals_policy import OBSERVATION_VERSION
    if env.metadata['observation_policy']['version'] != OBSERVATION_VERSION:
        raise EnvironmentError('Runtime observation version differs from the policy')
    placement = placement_options(policies)
    state = env.reset(map_path, seed=seed, max_frames=max_frames, opponent_kind=opponent_kind,
                      opponent_faction=opponent_faction, learning_side=learning_side,
                      record_replay=record_replay)
    map_features = state['map']
    if len(policies) != len(env.controlled_players):
        raise ValueError('Expected one policy per controlled player')
    for player, policy in zip(env.controlled_players, policies):
        policy.reset()
        policy.configure_clock(env.metadata['ticks_per_second'])
        # Policy opponents are vanilla China; a Hard AI's named faction is its lobby side.
        policy.configure_enemy('FactionChina' if opponent_kind == 'policy' else opponent_faction)
        if policy.policy.config['placement_mode'] == 'full_mask':
            policy.placement_query = lambda builder, template, angle, bins, player=player: env.placement(
                player, builder, template, angle, bins)
    stream = Path(trajectory).open('w', encoding='utf-8') if trajectory else None
    def write(record):
        if stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    try:
        write({'record': 'header', 'source': 'self_play', 'engine': env.metadata, 'map': map_features,
               'map_path': str(Path(map_path).resolve()), 'seed': seed, 'players': env.players,
               'execution_settings': placement, 'controlled_players': env.controlled_players,
               'setup': state.get('setup', {})})
        steps = 0
        while not env.done:
            actions = {p: policy.act(state['observations'][str(p)], map_features, state['frame'])
                       for p, policy in zip(env.controlled_players, policies)}
            advance = step_ticks(policies, state['frame'], ticks, env.metadata['ticks_per_second'])
            successor = env.step(actions, ticks=advance, placement=placement)
            for p, policy in zip(env.controlled_players, policies):
                policy.feedback([f for f in successor['action_feedback'] if f['player'] == p])
            write({'record': 'transition', 'frame': state['frame'], 'next_frame': successor['frame'],
                   'observations': state['observations'], 'actions': actions,
                   'next_observations': successor['observations'], 'rewards': successor['rewards'],
                   'terminated': successor['terminated'], 'truncated': successor['truncated'],
                   'action_feedback': successor['action_feedback']})
            state = successor
            steps += 1
        summary = {k: state[k] for k in ('frame', 'rewards', 'terminated', 'truncated')}
        summary.update(steps=steps, setup=state.get('setup', {}), execution_settings=placement,
                       event_scheduling={str(p): dict(policy.counters)
                                         for p, policy in zip(env.controlled_players, policies)})
        write({'record': 'footer', 'completed': True, **summary})
        return summary
    finally:
        if stream:
            stream.close()
