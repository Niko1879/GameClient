"""Event timing, causal selection, portable data and numerical continuation."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from event_policy import SCIENCE_ORDERS, EventActor
from event_schedule import delay_mask, native_deadline, step_ticks
from event_dataset import EventDataset
from event_store import feature_identity, write_shard
from generals_policy import NeuralPolicy
from nn_numpy import Tensor
from prepared_io import _write_json
from replay_events import decisions, encoded_events
from test_fixtures import fixture, raw_row
from train_events import EventLearner, train


class EventTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 5)

    def test_feature_identity_invalidates_after_any_feature_source_edit(self):
        import event_store
        digest, original = event_store._digest, feature_identity(self.policy)
        for name in event_store.FEATURE_SOURCES:
            with self.subTest(source=name), patch('event_store._digest',
                    side_effect=lambda path, name=name: 'edited' if Path(path).name == name else digest(path)):
                self.assertNotEqual(feature_identity(self.policy), original)
        getsource = event_store.inspect.getsource
        for obj in event_store.FEATURE_CODE:
            def edited(item, obj=obj):
                return getsource(item)+('\n# A subsequent edit.\n' if item is obj else '')
            # A process reads the feature code once; each new process sees an edit.
            with self.subTest(code=obj.__qualname__), patch('event_store.inspect.getsource', side_effect=edited), \
                    patch('event_store._FEATURE_CODE', []):
                self.assertNotEqual(feature_identity(self.policy), original)

    def test_ui_bookkeeping_does_not_censor_delay_or_reset_history(self):
        rows = [raw_row(0, self.obs), raw_row(5, self.obs, ui=True),
                raw_row(30, self.obs, [{'kind': 'move', 'units': [10], 'position': [200, 300, 0]}]),
                raw_row(37, self.obs, ui=True), raw_row(60, self.obs, [{'kind': 'stop', 'units': [10]}])]
        counts = Counter()
        chosen = list(decisions(rows, self.config, counts))
        self.assertEqual([r['source']['frame'] for r in chosen], [0, 30, 60])
        self.assertEqual([r['delay_ticks'] for r in chosen], [30, 30, None])
        self.assertEqual(chosen[2]['previous_frame'], 30)
        self.assertEqual(counts['ui_only_messages'], 2)

    def test_quiet_intervals_start_and_terminal_are_bounded_and_censored(self):
        rows = [raw_row(f, self.obs) for f in range(0, 307, 6)]
        rows[-1]['terminal_sample'] = True
        chosen = list(decisions(rows, self.config, Counter()))
        self.assertEqual([r['source']['frame'] for r in chosen], [0, 120, 240])
        self.assertEqual([r['delay_ticks'] for r in chosen], [120, 120, None])
        # Off-grid commands cannot cause a synthetic earlier observation.
        rows = [raw_row(1, self.obs, [{'kind': 'stop', 'units': [10]}])]
        rows += [raw_row(f, self.obs) for f in range(6, 259, 6)]
        selected = list(decisions(rows, self.config, Counter()))
        self.assertEqual(selected[1]['source']['frame'], 120)
        self.assertLessEqual(max(r['delay_ticks'] or 0 for r in selected), 120)

    def test_unsupported_events_are_boundaries_never_wait_labels(self):
        stop = [{'kind': 'stop', 'units': [10]}]
        rows = [raw_row(0, self.obs, stop), raw_row(40, self.obs, unsupported=True), raw_row(80, self.obs, stop)]
        counts = Counter()
        chosen = list(decisions(rows, self.config, counts))
        self.assertEqual([r['source']['frame'] for r in chosen], [0, 80])
        self.assertTrue(all(r['delay_ticks'] is None for r in chosen))
        self.assertEqual(counts['excluded:unsupported'], 1)

    def test_future_command_time_changes_targets_only(self):
        def convert(frame):
            rows = [raw_row(0, self.obs), raw_row(frame, self.obs, [{'kind': 'stop', 'units': [10]}])]
            game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2}
            return list(encoded_events(self.policy, game, Counter()))
        a, b = convert(30), convert(70)
        for name in ('scalars', 'numeric', 'extra_numeric', 'map_visibility'):
            np.testing.assert_array_equal(a[0]['encoded'][name], b[0]['encoded'][name])
        self.assertNotEqual(a[0]['target']['delay'], b[0]['target']['delay'])

    def test_structural_failure_censors_delay_without_committing_history(self):
        stop = [{'kind': 'stop', 'units': [10]}]
        bad = [{'kind': 'attack', 'units': [10], 'target': 123456}]
        rows = [raw_row(0, self.obs, stop), raw_row(30, self.obs, bad), raw_row(40, self.obs, stop)]
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2}
        counts = Counter()
        chosen = list(encoded_events(self.policy, game, counts))
        self.assertEqual([row['frame'] for row in chosen], [0, 40])
        self.assertNotIn('delay', chosen[0]['target'])
        self.assertEqual(counts['structural_exclusions'], 1)
        expected = self.policy.encoder.encode(self.obs, self.terrain, 40, previous_frame=0)
        np.testing.assert_array_equal(chosen[1]['encoded']['scalars'], expected['scalars'])

    def test_clock_rounding_multi_actor_deadlines_and_no_early_inference(self):
        actor = EventActor(self.policy, deterministic=True)
        actor.configure_clock(30)
        self.assertEqual(native_deadline(7, 30), 4)
        self.assertEqual([i+1 for i, enabled in enumerate(delay_mask(self.config, 30)) if enabled][:3], [2, 4, 6])
        trace = ([], {'kind': 'wait', 'delay': 7}, None, Tensor(0), Tensor(0), Tensor(0))
        with patch.object(self.policy, 'decide', return_value=trace), patch.object(
                self.policy.encoder, 'encode', wraps=self.policy.encoder.encode) as encoder:
            actor.act(self.obs, self.terrain, 0)
            self.assertEqual(step_ticks([actor], 3, 6, 30), 1)
            actor.act(self.obs, self.terrain, 3)
            self.assertEqual(encoder.call_count, 1)
            actor.act(self.obs, self.terrain, 4)
            self.assertEqual(encoder.call_count, 2)
        other = EventActor(self.policy)
        other.next_decision_frame = 5
        self.assertEqual(step_ticks([actor, other], 4, 6, 30), 1)

    def test_uncapped_events_retain_fast_commands_and_causal_history(self):
        stop = [{'kind': 'stop', 'units': [10]}]
        rows = [raw_row(0, self.obs), *[raw_row(f, self.obs, stop) for f in (1, 2, 3)],
                raw_row(6, self.obs)]
        counts = Counter()
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2}
        chosen = list(encoded_events(self.policy, game, counts))
        self.assertEqual([r['frame'] for r in chosen], [0, 1, 2, 3])
        self.assertEqual(counts['commands'], 3)
        self.assertEqual([r['target'].get('delay') for r in chosen], [0, 0, 0, None])
        expected = self.policy.encoder.encode(self.obs, self.terrain, 3, previous_frame=2)
        np.testing.assert_array_equal(chosen[-1]['encoded']['scalars'], expected['scalars'])

    def test_actor_conditions_on_a_rated_opponent_by_default(self):
        from generals_policy import PLAY_RATING, rating_inputs
        actor = EventActor(self.policy)
        rated = rating_inputs(PLAY_RATING, PLAY_RATING)
        # Per side (value, rated, estimated): live play conditions on two rated sides.
        self.assertEqual(rated[1::3].tolist(), [1, 1])
        self.assertEqual(rated[2::3].tolist(), [0, 0])
        np.testing.assert_array_equal(actor.ratings, rated)
        actor.configure_ratings(1500, None)
        np.testing.assert_array_equal(actor.ratings, rating_inputs(1500, None))
        actor.reset()  # run_episode resets every actor at game start
        np.testing.assert_array_equal(actor.ratings, rating_inputs(1500, None))
        actor.configure_ratings()
        np.testing.assert_array_equal(actor.ratings, rated)
        actor.reset()
        np.testing.assert_array_equal(actor.ratings, rated)
        trace = ([], {'kind': 'wait', 'delay': 7}, None, Tensor(0), Tensor(0), Tensor(0))
        # The rating input is off by default (config rating_input): the model then plays unrated.
        with patch.object(self.policy, 'decide', return_value=trace) as decide:
            actor.act(self.obs, self.terrain, 0)
        self.assertFalse(decide.call_args[0][0]['ratings'].any())
        actor.reset()
        with patch.object(self.policy, 'decide', return_value=trace) as decide, \
                patch.dict(self.policy.config, rating_input=True):
            actor.act(self.obs, self.terrain, 0)
        np.testing.assert_array_equal(decide.call_args[0][0]['ratings'], rated)
        # A model trained with GENERALS_RATING_INPUT=1 has no config key; its loader flags the policy instead.
        actor.reset()
        with patch.object(self.policy, 'decide', return_value=trace) as decide, \
                patch.object(self.policy, 'rating_input', True, create=True):
            actor.act(self.obs, self.terrain, 0)
        np.testing.assert_array_equal(decide.call_args[0][0]['ratings'], rated)

    def test_science_buyer_waits_for_the_next_science_and_never_skips(self):
        from event_policy import ScienceBuyer
        buyer = ScienceBuyer()
        orders = json.loads(SCIENCE_ORDERS.read_text(encoding='utf-8'))
        state = lambda owned, purchasable: {'player': {'sciences': owned, 'purchasable_sciences': purchasable}}
        buyer.configure('FactionGLA')
        gla = orders['orders']['FactionGLA']
        self.assertEqual(gla[0], 'Early_SCIENCE_ChinaCarpetBomb')
        # Rank 1 against GLA: Red Guard is purchasable but not first on the list, so the point is saved.
        self.assertEqual(buyer(state([], ['SCIENCE_RedGuardTraining', 'SCIENCE_NukeLauncher'])), [])
        self.assertEqual(buyer(state([], ['SCIENCE_RedGuardTraining', gla[0], gla[1]])),
                         [{'kind': 'science', 'science': gla[0]}])
        self.assertEqual(buyer(state(gla[:1], [gla[1], 'SCIENCE_CashHack1'])), [{'kind': 'science', 'science': gla[1]}])
        self.assertEqual(buyer(state(gla, [])), [])  # list done: nothing more is bought
        buyer.configure('FactionAmerica')
        self.assertEqual(buyer(state([], ['SCIENCE_RedGuardTraining'])),
                         [{'kind': 'science', 'science': 'SCIENCE_RedGuardTraining'}])
        for unknown in (None, 'random', 'FactionCivilian'):
            buyer.configure(unknown)
            self.assertEqual(buyer.order, orders['default'])
        # Every list is a vanilla China science the deployed vocabulary knows, and none repeats.
        # (Cloud bundles ship the orders but not the deployed policy.json; the vocabulary check runs locally.)
        deployed = SCIENCE_ORDERS.parent/'policy.json'
        known = set(json.loads(deployed.read_text(encoding='utf-8'))['sciences']) if deployed.exists() else None
        for order in [orders['default'], *orders['orders'].values()]:
            self.assertEqual(len(order), len(set(order)))
            if known is not None:
                self.assertLessEqual(set(order), known)

    def test_actor_buys_sciences_on_every_observation_with_the_model_actions(self):
        actor = EventActor(self.policy)
        actor.configure_enemy('FactionAmerica')
        obs = copy.deepcopy(self.obs)
        obs['player'].update(sciences=[], purchasable_sciences=['SCIENCE_RedGuardTraining'])
        buy = {'kind': 'science', 'science': 'SCIENCE_RedGuardTraining'}
        stop = {'kind': 'stop', 'units': [10]}
        trace = ([stop], {'kind': 'stop', 'delay': 29}, None, Tensor(0), Tensor(0), Tensor(0))
        with patch.object(self.policy, 'decide', return_value=trace):
            self.assertEqual(actor.act(obs, self.terrain, 0), [buy, stop])
            self.assertEqual(actor.act(obs, self.terrain, 5), [buy])  # a skipped inference still buys
            obs['player'].update(sciences=['SCIENCE_RedGuardTraining'], purchasable_sciences=[])
            self.assertEqual(actor.act(obs, self.terrain, 6), [])
        self.assertEqual(actor.counters['science_purchases'], 2)

    def test_uncapped_actor_keeps_learned_delay_after_acceptance_or_rejection(self):
        for delay in (1, 90):
            for accepted in (False, True):
                with self.subTest(delay=delay, accepted=accepted):
                    actor = EventActor(self.policy)
                    trace = ([{'kind': 'stop', 'units': [10]}],
                             {'kind': 'stop', 'delay': delay-1}, None, Tensor(0), Tensor(0), Tensor(0))
                    with patch.object(self.policy, 'decide', return_value=trace) as decide:
                        actor.act(self.obs, self.terrain, 0)
                        actor.feedback([{'accepted': accepted}])
                        self.assertEqual(actor.next_deadline_native, delay)
                        actor.act(self.obs, self.terrain, delay-1)
                        self.assertEqual(decide.call_count, 1)
                        actor.act(self.obs, self.terrain, delay)
                        self.assertEqual(decide.call_count, 2)

    def test_event_queues_are_uncapped_and_follow_engine_availability(self):
        producer = self.obs['own_entities'][1]
        producer['production_queue'] = [{'unit_type': 'ChinaVehicleDozer', 'progress': 0,
                                         'remaining_quantity': 4}]
        producer['trainable_templates'] = ['ChinaVehicleDozer']
        encoded = self.policy.encoder.encode(self.obs, self.terrain, 0)
        self.assertTrue(self.policy.actor_mask(encoded, 'train')[encoded['ids'][20]])
        self.policy.target(encoded, [{'kind': 'train', 'producer': 20, 'template': 'ChinaVehicleDozer'}])
        # Genuine availability still applies when the game says training is unavailable.
        producer['trainable_templates'] = []
        encoded = self.policy.encoder.encode(self.obs, self.terrain, 0)
        self.assertFalse(self.policy.actor_mask(encoded, 'train')[encoded['ids'][20]])

    def shard_dataset(self, root, *, validation=False):
        rows = [raw_row(0, self.obs)]
        rows += [raw_row(f, self.obs, [{'kind': 'stop', 'units': [10]}]) for f in range(30, 331, 30)]
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2}
        shard = root/'shards/a.zip'
        result = write_shard(shard, [game], self.policy, {'match_key': 'test:1', 'path': 'absent-export'})
        document = {'kind': 'generals_event_dataset', 'schema': 1, 'config': self.config,
                    'entries': [{**result, 'path': 'shards/a.zip', 'match_key': 'test:1', 'split': 'train'}]}
        if validation:
            heldout = write_shard(root/'shards/b.zip', [game], self.policy, {'match_key': 'test:2'})
            document['entries'].append({**heldout, 'path': 'shards/b.zip', 'match_key': 'test:2', 'split': 'validation'})
        _write_json(root/'dataset.json', document)
        return EventDataset(root/'dataset.json')

    def test_players_below_the_rating_floor_are_not_supervised(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(list(self.shard_dataset(root).chunks('train')))
            document = json.loads((root/'dataset.json').read_text())
            document['entries'][0]['excluded_players'] = ['2']
            _write_json(root/'dataset.json', document)
            self.assertEqual(list(EventDataset(root/'dataset.json').chunks('train')), [])

    def test_weapon_and_armour_stats_reach_every_row_from_any_observation(self):
        """Stats arrive once (re-export fix 4), here in an observation no training row uses;
        the episode table must still carry them to every stored row."""
        from generals_policy import DAMAGE_TYPES, arms_inputs
        armed = copy.deepcopy(self.obs)
        armed['own_entities'][0].update(weapons=[0, -1, -1], armor=0, weapon_bonus=[1.25, 1.33, 1.5])
        weapon = {'id': 0, 'name': 'TestGun', 'range': 150.0, 'min_range': 0.0, 'damage': 40.0, 'radius': 5.0,
                  'min_delay_frames': 30, 'max_delay_frames': 50, 'clip_size': 3, 'clip_reload_frames': 120,
                  'anti_mask': 5, 'damage_type': 2}
        delivered = copy.deepcopy(armed)
        delivered['template_stats'] = {'weapons': [weapon], 'armors': [{'id': 0, 'damage_multipliers': [.5]*DAMAGE_TYPES}]}
        rows = [raw_row(0, armed), raw_row(15, delivered)]
        rows += [raw_row(f, armed, [{'kind': 'stop', 'units': [10]}]) for f in range(30, 331, 30)]
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = write_shard(root/'shards/a.zip', [game], self.policy, {'match_key': 'test:1', 'path': 'absent'})
            _write_json(root/'dataset.json', {'kind': 'generals_event_dataset', 'schema': 1, 'config': self.config,
                'entries': [{**result, 'path': 'shards/a.zip', 'match_key': 'test:1', 'split': 'train'}]})
            dataset = EventDataset(root/'dataset.json')
            stored = [row for reference in dataset.chunks('train') for row in dataset.read(reference)]
        frames = [row['frame'] for row in stored]
        self.assertNotIn(15, frames)  # the delivering observation is not a training row
        self.assertGreater(len(stored), 5)
        for row in stored:
            encoded = row['encoded']
            np.testing.assert_allclose(encoded['weapon_stats'][0], [150, 0, 40, 5, 40/60, 3, 2, 5, 2])
            np.testing.assert_allclose(encoded['armor_stats'][0], [.5]*DAMAGE_TYPES)
            # Stored rows keep no id map; the armed dozer is the entity carrying weapon id 0.
            armed_row = int(np.flatnonzero(encoded['weapon_ids'][:, 0] == 0)[0])
            self.assertEqual(encoded['armor_ids'][armed_row], 0)
            arms = arms_inputs(encoded)[armed_row]
            self.assertEqual(arms[0], 1)  # primary weapon present
            self.assertAlmostEqual(arms[1], np.log1p(150*1.33)/7)  # range x range bonus
            self.assertAlmostEqual(arms[3], np.log1p(40*1.25)/7)  # damage x damage bonus
            self.assertAlmostEqual(arms[5], np.log1p((40/60)/1.5)/3)  # mean delay (s) / rate bonus

    def test_portable_columns_cursor_and_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.shard_dataset(root/'original')
            shutil.copytree(root/'original', root/'portable')
            shutil.rmtree(root/'original')
            data = EventDataset(root/'portable/dataset.json')
            all_batches = list(data.batches('train', seed=3, epoch=0, batch_size=3))
            resumed = list(data.batches('train', seed=3, epoch=0, batch_size=3, cursor=all_batches[0][1]))
            expected = [[r['frame'] for r in batch] for batch, _ in all_batches[1:]]
            self.assertEqual([[r['frame'] for r in batch] for batch, _ in resumed], expected)
            from contextlib import closing
            from event_training_batches import batches as prefetched
            with closing(prefetched(data, 'train', seed=3, epoch=0, batch_size=3,
                                   cursor=all_batches[0][1])) as stream:
                self.assertEqual([[r['frame'] for r in batch] for batch, _ in stream], expected)
            self.assertEqual(sum(len(batch) for batch, _ in all_batches), 12)
            rows = all_batches[0][0]
            learner = EventLearner(self.policy)
            learner.batch(rows, train=True)
            path = root/'portable/shards/a.zip'
            with path.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError):
                list(data.batches('train', seed=3, epoch=0))

    def test_reduced_precisions_need_cuda_and_take_finite_updates(self):
        encoded = self.policy.encoder.encode(self.obs, self.terrain, 0)
        rows = [{'encoded': encoded, 'target': {**self.policy.target(encoded, [{'kind': 'stop', 'units': [10]}]),
                                                'delay': 30}}]
        for precision in ('tf32', 'bf16', 'fp16'):
            with self.subTest(precision=precision):
                with self.assertRaises(ValueError):
                    EventLearner(NeuralPolicy(self.config, 5), precision=precision)
                if os.environ.get('TEST_TORCH_DEVICE') != 'cuda':
                    continue
                learner = EventLearner(NeuralPolicy(self.config, 5), device='cuda', precision=precision)
                totals = learner.batch(rows, train=True)
                self.assertTrue(np.isfinite(totals['loss_sum']))
                self.assertEqual(totals['updates'], 1)

    def test_optimizer_checkpoint_continuation_is_exact(self):
        encoded = self.policy.encoder.encode(self.obs, self.terrain, 0)
        rows = [{'encoded': encoded, 'target': {**self.policy.target(encoded, []), 'delay': 119}}]
        left = EventLearner(NeuralPolicy(self.config, 5), device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        right = EventLearner(NeuralPolicy(self.config, 5), device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        left.batch(rows, train=True); right.batch(rows, train=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.npz'
            right.save(path, {'cursor': [0, 1]})
            policy, optimizer, document = NeuralPolicy.load(path)
            right = EventLearner(policy, optimizer, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
            right.restore_rng(document['metadata'])
            # New command heads receive their first gradient after resuming.
            command = [{'encoded': encoded, 'target': {**self.policy.target(encoded,
                       [{'kind': 'stop', 'units': [10]}]), 'delay': 119}}]
            left.batch(command, train=True); right.batch(command, train=True)
            for name in left.model.weights:
                np.testing.assert_array_equal(left.model.weights[name].detach().cpu().numpy(),
                                              right.model.weights[name].detach().cpu().numpy())
                a, b = left.optimizer.state.get(left.model.weights[name]), right.optimizer.state.get(right.model.weights[name])
                if a:
                    self.assertEqual(a['step'].item(), b['step'].item())
                    np.testing.assert_array_equal(a['exp_avg'].cpu().numpy(), b['exp_avg'].cpu().numpy())

    def test_model_width_does_not_invalidate_features(self):
        config = copy.deepcopy(self.config); config['hidden'] *= 2
        self.assertEqual(feature_identity(self.policy), feature_identity(NeuralPolicy(config)))
        config['event_max_delay'], config['delay_ticks'] = 60, list(range(1, 61))
        self.assertNotEqual(feature_identity(self.policy), feature_identity(NeuralPolicy(config)))

    def test_low_memory_preflight_prevents_dataset_and_model_allocation(self):
        resources = {'free_disk_bytes': 100*1024**3, 'available_ram_bytes': 3*1024**3}
        with tempfile.TemporaryDirectory() as temporary, patch('train_events.snapshot', return_value=resources), \
                patch('train_events.EventDataset') as dataset:
            args = SimpleNamespace(output=temporary, min_free_ram_mib=4096, max_private_mib=8192)
            with self.assertRaisesRegex(RuntimeError, 'resource reserve'):
                train(args)
            dataset.assert_not_called()

    def test_training_loop_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch('train_events.within_budget', return_value=True):
            root = Path(temporary)
            data = self.shard_dataset(root/'data')
            settings = dict(dataset=str(data.path), resume=None, device='cpu', epochs=1, max_updates=0,
                seed=123, lr=3e-4, threads=1, batch_size=3, pair_budget=2_000_000,
                buffer_mib=1, checkpoint_seconds=600, min_free_ram_mib=4096, max_private_mib=8192)
            full = train(SimpleNamespace(**settings, output=str(root/'full')))
            stop_file = root/'stop.json'
            stop_file.write_text('{}')
            stopped = train(SimpleNamespace(**settings, output=str(root/'stopped'), stop_file=str(stop_file)))
            self.assertEqual(stopped['status'], 'interrupted')
            self.assertEqual(stopped['optimizer_updates'], 0)
            self.assertEqual(stopped['cursor'], {'epoch':0, 'position':[0,0], 'validations':0})
            stop_file.unlink()
            continued = train(SimpleNamespace(**{**settings, 'resume': stopped['checkpoint']},
                output=str(root/'stopped'), stop_file=str(stop_file)))
            self.assertEqual(continued['history'][0]['train'], full['history'][0]['train'])
            partial = train(SimpleNamespace(**{**settings, 'max_updates': 2}, output=str(root/'resumed')))
            self.assertEqual(partial['status'], 'update_limit')
            resumed = train(SimpleNamespace(**{**settings, 'resume': partial['checkpoint']}, output=str(root/'resumed')))
            self.assertEqual(resumed['status'], 'completed')
            self.assertEqual(full['history'][0]['train'], resumed['history'][0]['train'])
            self.assertEqual(full['optimizer_updates'], resumed['optimizer_updates'])
            a, a_opt, _ = NeuralPolicy.load(full['checkpoint'])
            b, b_opt, _ = NeuralPolicy.load(resumed['checkpoint'])
            self.assertEqual(a_opt.parameter_steps, b_opt.parameter_steps)
            for name in a.parameters:
                np.testing.assert_array_equal(a.parameters[name].data, b.parameters[name].data)
                np.testing.assert_array_equal(a_opt.m[name], b_opt.m[name])
                np.testing.assert_array_equal(a_opt.v[name], b_opt.v[name])

    def test_outliers_name_their_match_and_split_the_loss_by_head(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch('train_events.within_budget', return_value=True), patch('train_events.OUTLIER_NLL', -1.0):
            root = Path(temporary)
            data = self.shard_dataset(root/'data')
            result = train(SimpleNamespace(dataset=str(data.path), resume=None, device='cpu', epochs=1, max_updates=0,
                seed=123, lr=3e-4, threads=1, batch_size=3, pair_budget=2_000_000,
                buffer_mib=1, checkpoint_seconds=600, min_free_ram_mib=4096, max_private_mib=8192,
                output=str(root/'run')))
            records = [json.loads(line) for line in (root/'run'/'outliers.jsonl').read_text().splitlines()]
            history = result['history'][0]
            self.assertEqual(len(records), history['train']['decisions']+history['validation'].get('decisions', 0))
            self.assertEqual(history['train']['outliers'], history['train']['decisions'])
            keys = {entry['match_key'] for entry in data.entries}
            for record in records:
                self.assertIn(record['match_key'], keys)
                self.assertIn(record['split'], ('train', 'validation'))
                self.assertIsNotNone(record['player'])
                self.assertAlmostEqual(sum(record['terms'].values()), record['nll'], places=4)
                self.assertIn('kind', record['target'])

    def test_outlier_recheck_blames_corrupted_staged_inputs(self):
        from training_preparation import prepare_update
        with tempfile.TemporaryDirectory() as temporary, patch('train_events.OUTLIER_NLL', -1.0):
            data = self.shard_dataset(Path(temporary))
            rows = next(data.batches('train', seed=1, epoch=0, batch_size=6))[0]
            learner = EventLearner(NeuralPolicy(data.config, 5))
            learner.batch(rows, train=False)
            clean = {(r['frame'], r['player']): r for r in learner.outliers}
            learner.outliers.clear()
            # The device copy differs from the host's, as a broken upload would leave it.
            prepared = prepare_update(learner.model, rows, learner.pair_budget)
            for part in prepared['microbatches']:
                part['staged_values'] = learner.model.stager.stage(
                    {name: (value*50 if kind == 'float' else value, kind) for name, (value, kind) in part['values'].items()})
            learner.run(prepared, train=False)
            self.assertEqual(len(learner.outliers), len(clean))
            moved = 0
            for record in learner.outliers:
                reference = clean[(record['frame'], record['player'])]
                self.assertAlmostEqual(record['recheck']['unstaged'], reference['nll'], places=4)
                self.assertAlmostEqual(record['recheck']['fp32'], reference['nll'], places=4)
                self.assertFalse(record['after_update'])
                moved += abs(record['nll']-reference['nll']) > 1e-3
            self.assertGreater(moved, 0)

    def test_init_starts_from_checkpoint_weights_with_a_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                patch('train_events.within_budget', return_value=True):
            root = Path(temporary)
            data = self.shard_dataset(root/'data')
            settings = dict(dataset=str(data.path), resume=None, device='cpu', epochs=1, max_updates=0,
                seed=123, lr=3e-4, threads=1, batch_size=3, pair_budget=2_000_000,
                buffer_mib=1, checkpoint_seconds=600, min_free_ram_mib=4096, max_private_mib=8192)
            base = train(SimpleNamespace(**settings, output=str(root/'base')))
            tuned = train(SimpleNamespace(**{**settings, 'lr': 1e-4, 'max_updates': 1, 'init': base['checkpoint'],
                                             'select_on': '1400+'}, output=str(root/'tuned')))
            start, start_opt, document = NeuralPolicy.load(root/'tuned'/'initial.npz')
            trained, _, _ = NeuralPolicy.load(base['checkpoint'])
            self.assertEqual(start_opt.step_count, 0)
            self.assertIn('init_sha256', document['metadata']['event_settings'])
            for name in trained.parameters:
                np.testing.assert_array_equal(start.parameters[name].data, trained.parameters[name].data)
            self.assertEqual(tuned['optimizer_updates'], 1)
            # --validate-only scores the --init weights once, with no update; by_general splits the score.
            held = self.shard_dataset(root/'held', validation=True)
            scored = train(SimpleNamespace(**{**settings, 'dataset': str(held.path), 'init': base['checkpoint'], 'validate_only': True,
                                              'select_on': 'vanilla'}, output=str(root/'scored')))
            self.assertEqual((scored['status'], scored['optimizer_updates'], len(scored['history'])), ('validated', 0, 1))
            validation = scored['history'][0]['validation']
            self.assertGreater(validation['decisions'], 0)
            self.assertEqual(sum(g['decisions'] for g in validation['by_general'].values()), validation['decisions'])
            self.assertEqual(list(validation['by_general']), ['vanilla'])
            self.assertAlmostEqual(scored['history'][0]['selection_score'], validation['by_general']['vanilla']['mean_nll'])
            with self.assertRaises(ValueError):
                train(SimpleNamespace(**{**settings, 'dataset': str(held.path), 'validate_only': True}, output=str(root/'noinit')))
            # Resuming a fine-tune needs the same --init.
            with self.assertRaises(ValueError):
                train(SimpleNamespace(**{**settings, 'lr': 1e-4, 'select_on': '1400+',
                                         'resume': tuned['checkpoint']}, output=str(root/'tuned')))
            document = json.loads(Path(data.path).read_text())
            document['config'] = {**document['config'], 'hidden': document['config']['hidden']*2}
            wider = Path(data.path).with_name('wider.json')
            wider.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                train(SimpleNamespace(**{**settings, 'dataset': str(wider), 'init': base['checkpoint']},
                                      output=str(root/'wider')))


class MicrobatchTests(unittest.TestCase):
    def test_microbatches_cover_each_row_once_within_pair_budget(self):
        from train_events import microbatches
        from training_preparation import MICROBATCH_ROWS
        sizes = np.random.default_rng(3).integers(1, 60, 200)
        rows = [{'id': str(i), 'encoded': {'numeric': np.zeros((int(size), 1))}} for i, size in enumerate(sizes)]
        rows.append({'id': 'large', 'encoded': {'numeric': np.zeros((120, 1))}})
        groups = list(microbatches(rows, 4000))
        self.assertEqual(sorted(r['id'] for g in groups for r in g), sorted(r['id'] for r in rows))
        for group in groups:
            largest = max(len(r['encoded']['numeric']) for r in group)
            self.assertTrue(len(group) == 1 or len(group)*largest*largest <= 4000)
        self.assertEqual([len(g) for g in groups if any(r['id'] == 'large' for r in g)], [1])
        self.assertEqual(list(microbatches(rows[:1], 4000)), [rows[:1]])
        # Small observations under a generous budget: rows are capped, not only pairs (v11 big OOM, 17 Nov).
        small = [{'id': str(i), 'encoded': {'numeric': np.zeros((20, 1))}} for i in range(512)]
        groups = list(microbatches(small, 8_000_000))
        self.assertEqual(sum(map(len, groups)), 512)
        self.assertLessEqual(max(map(len, groups)), MICROBATCH_ROWS)


if __name__ == '__main__':
    unittest.main()
