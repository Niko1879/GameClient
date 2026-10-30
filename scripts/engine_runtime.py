"""Private engine profiles and handshake preflight, so concurrent engines never share user data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from replay_training import require

PROFILE_ENV = 'GENERALS_AI_USER_DATA'


def clone_file(source, destination):
    # Engines never modify installed BIG/EXE/DLL/SCB/BIK assets. Writable small files
    # get independent copies; log/output/cache files are not shared hardlinks.
    if Path(source).suffix.lower() in ('.big', '.exe', '.dll', '.scb', '.bik'):
        try:
            return os.link(source, destination)
        except OSError:
            pass
    return shutil.copy2(source, destination)


def disable_power_throttling(handle=None):
    """Opt a process out of Windows EcoQoS (default: this process).

    Windows 11 throttles windowless background processes and, on a hybrid CPU,
    confines them to the efficiency cores: 18 exports shared the i7-12700's
    4 E-cores while all 16 P-core threads idled (one export took 57 s instead
    of 23 s). The opt-out is not inherited, so each engine needs its own.
    """
    if os.name != 'nt':
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    # ProcessPowerThrottling (4): control EXECUTION_SPEED (1) and turn it off.
    state = State(1, 1, 0)
    if not kernel.SetProcessInformation(kernel.GetCurrentProcess() if handle is None else handle,
                                        4, ctypes.byref(state), ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())


def isolation_profile(source, destination, maps=None):
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'Replays').mkdir(exist_ok=True)
    if source:
        source = Path(source).resolve()
        if (source/'Maps').is_dir():
            # Seed just the selected custom map directories in production.
            # Avoid copying a user's complete downloaded map collection.
            requested = None if maps is None else {part.casefold() for name in maps
                for part in str(name).replace('\\', '/').split('/') if part}
            target_maps = destination/'Maps'
            for path in (source/'Maps').iterdir():
                if requested is not None and path.name.casefold() not in requested:
                    continue
                target = target_maps/path.name
                target_maps.mkdir(exist_ok=True)
                if not target.exists():
                    if path.is_dir():
                        shutil.copytree(path, target, copy_function=clone_file)
                    else:
                        clone_file(path, target)
        for name in ('GeneralsOnlineData', 'GeneralsOnlineGameData'):
            if (source/name).is_dir():
                for asset in (source/name).rglob('*'):
                    if asset.is_file() and asset.suffix.lower() in ('.big', '.scb'):
                        target = destination/name/asset.relative_to(source/name)
                        if not target.exists():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            clone_file(asset, target)
        for path in source.glob('*.big'):
            target = destination/path.name
            if not target.exists():
                clone_file(path, target)
    return destination


def preflight(executable, profile, log, timeout):
    """Prove private-profile support and a ready handshake before concurrent use."""
    environment = {**os.environ, PROFILE_ENV: str(profile)}
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.run([str(executable), '-headless', '-selfPlay', '-noshellmap'],
            cwd=Path(executable).parent, env=environment,
            input=json.dumps({'request_id': 1, 'command': 'close'})+'\n', text=True,
            stdout=output, stderr=subprocess.STDOUT, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    text = log.read_text(encoding='utf-8', errors='replace')
    profiles = [line[len('GeneralsAIProfile '):].rstrip('\\/') for line in text.splitlines()
                if line.startswith('GeneralsAIProfile ')]
    require(process.returncode == 0, 'Isolation preflight failed: '+str(log))
    require(profiles and all(Path(path).resolve() == profile.resolve() for path in profiles),
            'Executable does not prove private-profile support: '+str(executable))
    ready = [json.loads(line[4:]) for line in text.splitlines() if line.startswith('GAI ')]
    require(any(row.get('event') == 'ready' for row in ready), 'No engine handshake during isolation preflight')
