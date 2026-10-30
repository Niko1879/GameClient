"""Qualification retries must not conceal failures or override newer controls."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_event_training as profile


class QualificationQueueTests(unittest.TestCase):
    def run_suite(self, root, child, *, changed_control=False):
        (root/'scripts').mkdir()
        (root/'scripts/runtime.py').write_text('pass\n')
        (root/'replay-runtime').mkdir()
        control = root/'replay-runtime/converter.json'
        control.write_text('{"workers":4}')
        dataset = root/'dataset.json'
        dataset.write_text('{}')
        output = root/'profile'
        output.mkdir()
        args = SimpleNamespace(output=output, dataset=dataset, restore_converter_workers=4,
            pair_budget=8_000_000, batch_size=256, buffer_mib=64, threads=1,
            device='cuda', seed=123)

        def execute(command, **kwargs):
            if '--label' in command:
                label = command[command.index('--label')+1]
                code = child(label)
                if code == 0:
                    profile.write(output/(label+'.json'), dict(
                        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                        measured_seconds=1.0, examples_per_second=10.0,
                        counts={'decisions': 10}, resources={}))
                    if changed_control:
                        control.write_text('{"workers":2,"paused":true}')
                return subprocess.CompletedProcess(command, code)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(profile, 'ROOT', root), \
                patch('replay_converter.InstanceLock', return_value=nullcontext()), \
                patch('replay_converter.is_running', return_value=False), \
                patch.object(profile.subprocess, 'run', side_effect=execute) as run, \
                patch.object(profile.time, 'sleep') as sleep:
            profile.confirmation_suite(args)
        return json.loads((output/'confirmation-status.json').read_text()), run.call_args_list, sleep

    def test_only_resource_failures_retry_and_success_restores_converter(self):
        attempts = []

        def child(label):
            attempts.append(label)
            return 75 if len(attempts) == 1 else 0

        with tempfile.TemporaryDirectory() as temporary:
            state, calls, sleep = self.run_suite(Path(temporary), child)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['attempts']['final-confirm'], 2)
        self.assertEqual(len(state['results']), 4)
        self.assertEqual(state['restored_converter_workers'], 4)
        sleep.assert_called_once_with(15)
        self.assertEqual(calls[-2].args[0][-4:], ['set', '--workers', '4', '--resume'])
        self.assertEqual(calls[-1].args[0][-1], 'start')

    def test_permanent_failure_stops_without_starting_converter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_suite(root, lambda label: 2)
            state = json.loads((root/'profile/confirmation-status.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['attempts'], {'final-confirm': 1})
            self.assertNotIn('restored_converter_workers', state)

    def test_newer_user_controls_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, calls, _ = self.run_suite(Path(temporary), lambda label: 0, changed_control=True)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['converter_restoration'], 'skipped_newer_user_control')
        self.assertTrue(all('--label' in call.args[0] for call in calls))


if __name__ == '__main__':
    unittest.main()
