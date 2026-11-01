"""Scheduled quiet capture: export only the quiet observations preparation can use.

With `-trainingQuietCapture scheduled` the engine writes every sample's metadata,
but a quiet cadence sample carries its observation only when this controller
names its frame; command samples and each player's first sample always carry
one. The controller follows the export file as it is written and runs the same
event schedule as preparation (replay_events.decisions with EventValidator,
which encodes command rows to learn whether they are accepted). The engine flushes
every row with an observation and says so ("rows"), so validation runs alongside
the game. At a quiet cadence frame it may ask which quiet frames will need one:

  no decision yet         -> this frame
  last decision at L      -> the last quiet frame before L+cap and the first at
                             or after L+cap: the only quiet rows decisions() can
                             select before another observed row changes L
  episode finished        -> none

decisions() changes its last decision only while consuming an observed row, and
the rows an answer names cannot change it before its last frame, so an answer
stays valid until then or until a command row. Commands only move the schedule
later, so a quiet frame before the first frame of the latest answer cannot be
selected and the engine does not ask there. Future quiet frames come from the
original replay's command stream (audit_replay_commands.binary_commands, which
every export is reconciled against); each arriving row is checked against it.
Preparation still reads the finished export; a quiet row it needs but cannot
find fails loudly rather than producing different data.
"""
from __future__ import annotations

from collections import deque
from itertools import chain
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from audit_replay_commands import binary_commands, message_names
from binary_replay import stream_records
from engine_runtime import disable_power_throttling
from generals_policy import NeuralPolicy
from replay_actions import ReplayActionAdapter
from replay_events import EventValidator, decisions

IGNORED = {'MSG_BEGIN_NETWORK_MESSAGES', 'MSG_LOGIC_CRC', 'MSG_SET_REPLAY_CAMERA'}
ROWS_READY = 'GAIX {"event":"rows"}\n'


def command_frames(replay):
    """Frames with policy messages, per player, exactly as reconcile() counts them."""
    names, frames = message_names(), {}
    for frame, kind, player in binary_commands(replay):
        if kind in names and names[kind] not in IGNORED:
            frames.setdefault(player, set()).add(frame)
    return frames


def wanted_frames(frame, last_decision, finished, cap, cadence, quiet):
    """Quiet frames from `frame` on that decisions() can select from this state.

    From last decision L, a quiet row is selected either as the last quiet row
    before L+cap (promoted when the next row passes L+cap) or as the first
    quiet row at or after L+cap. With no decision yet, the next quiet row is.
    """
    if finished:
        return []
    if last_decision is None:
        return [frame]
    end, wanted = last_decision+cap, []
    last = (end-1)//cadence*cadence
    while last >= frame and not quiet(last):
        last -= cadence
    if last >= frame:
        wanted.append(last)
    first = max(frame, -(-end//cadence)*cadence)
    while not quiet(first):
        first += cadence
    wanted.append(first)
    return wanted


class GrowingFile:
    """Blocking reads of a file another process is still writing.

    Opened with delete sharing, so the engine can rename its .part file when it
    finishes. A short read happens only after `finished` is set. Records are small
    and many, so reads ahead keep most of them from needing a system call, and
    from giving up the GIL to a schedule thread every record.
    """
    READ_AHEAD = 256*1024

    def __init__(self, path, finished):
        self.path, self.finished, self.offset = Path(path), finished, 0
        self.more, self.stream = threading.Event(), None
        self.buffer, self.position = b'', 0

    def _open(self):
        while self.stream is None:
            if self.path.exists():
                self.stream = open_shared(self.path)
            elif self.finished.is_set():
                raise FileNotFoundError(str(self.path))
            else:
                time.sleep(.05)

    def read(self, size):
        self._open()
        start, stop = self.position, self.position+size
        if stop <= len(self.buffer):
            self.position = stop
            self.offset += size
            return self.buffer[start:stop]
        data = self.buffer[start:]
        self.buffer, self.position = b'', 0
        while len(data) < size:
            chunk = self.stream.read(self.READ_AHEAD)
            if not chunk and self.finished.is_set():
                chunk = self.stream.read(self.READ_AHEAD)
                if not chunk:
                    break
            if not chunk:
                self.more.wait(.05)
                self.more.clear()
                continue
            needed = size-len(data)
            data += chunk[:needed]
            if len(chunk) > needed:
                self.buffer = chunk
                self.position = needed
        self.offset += len(data)
        return data

    def tell(self):
        return self.offset

    def close(self):
        if self.stream is not None:
            self.stream.close()


def open_shared(path):
    if os.name != 'nt':
        return open(path, 'rb', buffering=0)
    import ctypes
    import msvcrt
    from ctypes import wintypes
    create = ctypes.windll.kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    # GENERIC_READ; share read/write/delete; OPEN_EXISTING; normal attributes.
    handle = create(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), 'Cannot open export for scheduled capture', str(path))
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), 'rb', buffering=0)


class PlayerSchedule(threading.Thread):
    """One player's decisions() run over rows as they arrive."""
    def __init__(self, player, policy, game):
        super().__init__(name=f'capture-schedule-{player}', daemon=True)
        self.player, self.policy, self.game = player, policy, game
        self.rows, self.condition = deque(), threading.Condition()
        self.waiting = self.closed = self.done = False
        self.last_decision, self.finished, self.error = None, False, None
        self.counts = {}

    def put(self, row):
        with self.condition:
            self.rows.append(row)
            if self.waiting:
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def _source(self):
        while True:
            with self.condition:
                while not self.rows and not self.closed:
                    self.waiting = True
                    self.condition.notify_all()
                    self.condition.wait()
                self.waiting = False
                if not self.rows:
                    return
                row = self.rows.popleft()
            yield row

    def observe(self, last_decision, finished):
        self.last_decision, self.finished = last_decision, finished

    def run(self):
        from collections import Counter
        self.counts = Counter()
        try:
            rows = self._source()
            first = next(rows, None)
            if first is None:
                return
            validate = EventValidator(self.policy, self.game, self.counts, first['observation'], schedule_only=True)
            for _ in decisions(chain([first], rows), self.policy.config, self.counts,
                               validate=validate, observe=self.observe):
                pass
        except BaseException as error:
            self.error = error
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def settled(self, timeout):
        """Wait until every queued row is consumed; the schedule state is then current."""
        deadline = time.monotonic()+timeout
        with self.condition:
            while not self.done and not (self.waiting and not self.rows):
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Capture schedule did not settle')
                self.condition.wait(remaining)
        if self.error is not None:
            raise RuntimeError(f'Capture schedule for player {self.player} failed') from self.error


class CaptureController:
    def __init__(self, policy_path, replay, trajectory):
        config = json.loads(Path(policy_path).read_text(encoding='utf-8'))
        self.policy = NeuralPolicy(config, parameters=False)
        self.cap = config['event_max_delay']
        self.commands = command_frames(replay)
        self.finished = threading.Event()
        self.file = GrowingFile(Path(str(trajectory)+'.part'), self.finished)
        self.header = self.terrain = None
        self.players, self.adapters, self.last_frame = {}, {}, {}
        self.cadence, self.error = None, None
        self.dispatched = 0  # bytes whose records have reached their player schedules
        self.ready, self.asking = threading.Condition(), False
        self.reader = threading.Thread(target=self._read, name='capture-reader', daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for record in stream_records(self.file, lazy=True, validated=True):
                kind = record['record']
                if kind == 'header':
                    if record.get('quiet_capture') != 'scheduled' or record['ticks_per_second'] != 60:
                        raise ValueError('Scheduled capture needs a 60 Hz scheduled-capture export')
                    self.header = record
                    self.cadence = max(1, (record['interval_ms']*record['ticks_per_second']+999)//1000)
                elif kind == 'map':
                    self.terrain = record
                elif kind == 'sample':
                    self._sample(record)
                elif kind == 'footer':
                    break
                self.dispatched = self.file.offset
                if self.asking:  # set under the lock before directive() reads dispatched
                    with self.ready:
                        self.ready.notify_all()
        except BaseException as error:
            self.error = error
        finally:
            for schedule in self.players.values():
                schedule.close()
            with self.ready:
                self.ready.notify_all()

    def _sample(self, record):
        player, frame = record['player_index'], record['frame']
        commanded = frame in self.commands.get(player, ())
        if commanded != bool(record['actions']):
            raise ValueError(f'Replay command stream disagrees with export at frame {frame}')
        if record.get('light') and (commanded or frame % self.cadence):
            raise ValueError('Light sample outside a quiet cadence frame')
        if player not in self.players:
            self.adapters[player] = ReplayActionAdapter(self.header['template_catalog'])
            game = {'map': self.terrain, 'source_ticks_per_second': self.header['ticks_per_second'], 'player': player}
            self.players[player] = PlayerSchedule(player, self.policy, game)
            self.players[player].start()
        # The same demonstration fields iter_demonstrations gives decisions().
        row = {'record': 'demonstration', 'source': 'replay', 'player_index': player, 'frame': frame,
               **self.adapters[player].decode(record), 'reward': None}
        if 'observation' in record:
            row['observation'] = record['observation']
        self.last_frame[player] = frame
        self.players[player].put(row)

    def quiet(self, player, frame):
        return frame % self.cadence == 0 and frame not in self.commands.get(player, ())

    def directive(self, player, frame, timeout=600):
        """Quiet frames from `frame` on whose observations decisions() may select."""
        # The engine flushed every earlier row before asking: consume them all.
        size = self.file.path.stat().st_size if self.file.path.exists() else 0
        deadline = time.monotonic()+timeout
        self.file.more.set()
        with self.ready:
            self.asking = True
            try:
                while self.dispatched < size and self.error is None:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Scheduled capture reader did not catch up')
                    self.ready.wait(remaining)
            finally:
                self.asking = False
        if self.error is not None:
            raise RuntimeError('Scheduled capture reader failed') from self.error
        schedule = self.players.get(player)
        if schedule is None or not self.quiet(player, frame) or self.last_frame[player] >= frame:
            raise ValueError(f'Unexpected capture request for player {player} at frame {frame}')
        schedule.settled(timeout)
        return wanted_frames(frame, schedule.last_decision, schedule.done or schedule.finished,
                             self.cap, self.cadence, lambda f: self.quiet(player, f))

    def close(self):
        self.finished.set()
        self.file.more.set()
        self.reader.join(60)
        for schedule in self.players.values():
            schedule.close()
            schedule.join(60)
        self.file.close()
        if self.error is not None:
            raise RuntimeError('Scheduled capture reader failed') from self.error
        for schedule in self.players.values():
            if schedule.error is not None:
                raise RuntimeError(f'Capture schedule for player {schedule.player} failed') from schedule.error


def _relay(process, log, controller):
    """Copy engine output to `log`, answering capture requests.

    Returns a -replayServer job's exit code, or None when the output ends.
    """
    for line in process.stdout:
        if not line.startswith('GAIX '):
            log.write(line)
            continue
        if line == ROWS_READY:
            controller.file.more.set()
            continue
        request = json.loads(line[5:])
        if request.get('event') == 'replay_done':
            return request['exit_code']
        if request.get('event') != 'capture':
            raise ValueError('Unknown engine request: '+line.strip())
        frames = controller.directive(request['player'], request['frame'])
        process.stdin.write(json.dumps({'player': request['player'], 'frame': request['frame'],
                                        'frames': frames})+'\n')
        process.stdin.flush()
    return None


def _close_quietly(controller):
    try:
        if controller is not None:
            controller.close()
    except Exception:
        pass  # the original failure is the one to report


def _start_engine(command, cwd, creationflags):
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, creationflags=creationflags,
                               text=True, encoding='utf-8', errors='replace', bufsize=1)
    try:
        disable_power_throttling(int(process._handle))
    except BaseException:
        process.kill()
        process.wait()
        raise
    return process


def run_engine(command, *, cwd, log_path, timeout, creationflags, controller=None):
    """Run the native export, answering capture requests; returns the exit code.

    Engine output goes to `log_path` as before; capture requests stay off it.
    Raises subprocess.TimeoutExpired after killing the engine.
    """
    if controller is None:
        with Path(log_path).open('w', encoding='utf-8') as log:
            return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, creationflags=creationflags).returncode
    process = _start_engine(command, cwd, creationflags)
    expired = threading.Event()

    def watchdog():
        if not finished.wait(timeout):
            expired.set()
            process.kill()

    finished = threading.Event()
    threading.Thread(target=watchdog, name='capture-watchdog', daemon=True).start()
    try:
        with Path(log_path).open('w', encoding='utf-8') as log:
            _relay(process, log, controller)
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        finished.set()
        _close_quietly(controller)
        raise
    finished.set()
    if expired.is_set():
        _close_quietly(controller)
        raise subprocess.TimeoutExpired(command, timeout)
    controller.close()
    return returncode


class EngineSession:
    """One reusable -replayServer engine, restarted when its command changes.

    A replay's INI package is fixed at engine startup, so a new package needs a
    new process. `max_games` bounds any slow growth in a long-lived process.
    """
    def __init__(self, max_games=100):
        self.max_games, self.process, self.command, self.games = max_games, None, None, 0

    def live(self):
        return self.process is not None and self.process.poll() is None

    def close(self):
        process, self.process, self.games = self.process, None, 0
        if process is None:
            return
        try:
            process.communicate(timeout=30)  # end of stdin ends the engine
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
            process.communicate()

    def _kill(self):
        process, self.process, self.games = self.process, None, 0
        process.kill()
        process.communicate()

    def run(self, command, job, *, cwd, log_path, timeout, creationflags, controller=None):
        """Export one job; returns (exit code, earlier games this process played)."""
        if (self.process is None or self.process.poll() is not None or self.command != command
                or self.games >= self.max_games):
            self.close()
            self.process, self.command = _start_engine(command, cwd, creationflags), list(command)
        earlier, process = self.games, self.process
        expired = threading.Event()

        def expire():
            expired.set()
            process.kill()

        watchdog = threading.Timer(timeout, expire)
        watchdog.daemon = True
        watchdog.start()
        try:
            process.stdin.write(json.dumps(job)+'\n')
            process.stdin.flush()
            with Path(log_path).open('w', encoding='utf-8') as log:
                code = _relay(process, log, controller)
        except BaseException:
            self._kill()
            _close_quietly(controller)
            raise
        finally:
            watchdog.cancel()
        if expired.is_set():
            self._kill()
            _close_quietly(controller)
            raise subprocess.TimeoutExpired(command, timeout)
        if code is None:  # the engine ended mid-job
            code = process.wait() or 1
            self._kill()
        else:
            self.games += 1
        if controller is not None:
            controller.close()
        return code, earlier
