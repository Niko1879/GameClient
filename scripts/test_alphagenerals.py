"""AlphaGenerals (notes/plan.md §1-2): the NumPy reference and the batched learner agree on every
head; the flat action covers every stored target; the LSTM steps the same live and in training;
the lane sampler visits every decision once, in order; checkpoints round-trip; old checkpoints play."""
import copy
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from event_store import feature_identity, write_shard
from generals_policy import COMMAND_KINDS, KINDS, NeuralPolicy, action_object
from nn_numpy import no_grad
from prepared_io import _write_json
from tensor_replay_store import TensorRows, decode_columns, encode_columns, project
from test_command_actions import ACTIONS as COMMAND_ACTIONS, fixture_commands
from test_current_perception import perceived
from test_fixtures import RECORDING, entity, fixture, raw_row
from test_policy import ACTIONS as BASIC_ACTIONS
from torch_policy_v3 import TorchPolicyV3
from torch_training_inputs import teacher_heads
from train_events import EventLearner
from training_legality import build_legality
from training_preparation import encode_values, teacher_plan

ROOT = Path(__file__).resolve().parents[1]
OLD_CHECKPOINTS = (ROOT/'.cache/cloud-runs/v11-1500/best.npz', ROOT/'.cache/cloud-runs/v11-70m/best.npz')


def observations():
    """(name, config, observation, terrain): commands without effects and with an empty neutral
    group; with v11 effects and sound cues; and with a neutral unit."""
    config, obs, terrain = fixture_commands()
    with_effects = perceived(copy.deepcopy(obs))
    neutral = copy.deepcopy(with_effects)
    neutral['known_nonowned_entities'] += [entity(200, 'Enemy', False, [600, 300, 0], owner=0),
                                           entity(201, 'Enemy', False, [650, 320, 0], owner=1)]
    return [('plain', config, obs, terrain), ('effects', config, with_effects, terrain), ('neutral', config, neutral, terrain)]


def rows_for(policy, observation, terrain):
    """One teacher row per action of the command and basic fixtures that this observation can express."""
    rows = []
    for actions in ([], *COMMAND_ACTIONS, *BASIC_ACTIONS[1:]):
        encoded = policy.encoder.encode(observation, terrain, 6)
        try:
            target = policy.target(encoded, actions)
        except ValueError:  # e.g. a basic group order naming units this fixture does not hold
            continue
        rows.append({'encoded': encoded, 'target': target})
    return rows


def numpy_logp(policy, row, memory=None):
    with no_grad():
        result = policy.decide(row['encoded'], memory, teacher=row['target'])
    return float(result[4].data), result[2]


class HeadAgreementTests(unittest.TestCase):
    def test_numpy_and_torch_agree_on_every_head_with_and_without_effects_and_neutrals(self):
        heads = set()
        for name, config, observation, terrain in observations():
            policy = NeuralPolicy(config, 5)
            rows = rows_for(policy, observation, terrain)
            model = TorchPolicyV3(policy, dtype=torch.float64)
            encoded = [row['encoded'] for row in rows]
            output = model.teacher_heads(encoded, [row['target'] for row in rows], *model.events(encoded))
            expected = [numpy_logp(policy, row)[0] for row in rows]
            with self.subTest(name):
                np.testing.assert_allclose(output['logp'].detach().numpy(), expected, rtol=1e-9, atol=1e-9)
            heads |= {head for head, counts in output['head_counts'].items() if counts.any()}
            if name == 'plain':
                self.assertFalse(any(e['owner'] < 2 and not e['own'] for e in encoded[0]['entities']))
                self.assertEqual(len(encoded[0]['effect_numeric']), 0)
            else:
                self.assertGreater(len(encoded[0]['effect_numeric']), 0)
        self.assertEqual(heads, {'decision', 'unit', 'target', 'slot', 'angle', 'cell', 'offset'})

    def test_full_mask_build_scores_position_with_its_placement_features(self):
        from test_build_placement import PlacementTests
        case = PlacementTests('setUp'); case.setUp()
        rows = [case.example(), case.example(angle=7)]
        model = TorchPolicyV3(case.policy, dtype=torch.float64)
        encoded = [row['encoded'] for row in rows]
        output = model.teacher_heads(encoded, [row['target'] for row in rows], *model.events(encoded))
        np.testing.assert_allclose(output['logp'].detach().numpy(), [numpy_logp(case.policy, row)[0] for row in rows],
                                   rtol=1e-9, atol=1e-9)


class FlatActionTests(unittest.TestCase):
    def test_every_stored_target_is_one_unmasked_vocabulary_entry(self):
        for name, config, observation, terrain in observations():
            policy = NeuralPolicy(config, 5)
            rows = rows_for(policy, observation, terrain)
            key = feature_identity(policy)
            for row in rows:
                row['encoded']['_training_legality'] = build_legality(policy, row['encoded'], row['target'], key)
            # Through the stored columns, as training reads them: the plan's categorical check covers every row.
            packed = project(rows, {}, policy, key)
            portable = list(TensorRows(*decode_columns(encode_columns(packed.columns, packed.metadata)), {}, key))
            model = TorchPolicyV3(policy, dtype=torch.float64)
            model.legality_key = key
            plan = teacher_plan(model, [row['encoded'] for row in portable], [row['target'] for row in portable], 64)
            choices = plan['values']['decision_choices'][0]
            for row, choice in zip(portable, choices):
                kind, obj = policy.actions[choice]
                self.assertEqual((KINDS[kind], int(obj)), (row['target']['kind'], action_object(row['target'])))
                self.assertTrue(policy.action_mask(row['encoded'], row['encoded']['_training_legality']['verb'])[choice])
            for row in rows:
                encoded = row['encoded']
                mask, exact = policy.action_mask(encoded), policy.exact_action_mask(encoded)
                self.assertFalse((exact & ~mask).any(), 'exact choices are a subset of the stored superset')
                np.testing.assert_array_equal(mask[policy.actions[:, 1] < 0], policy.verb_mask(encoded)[policy.actions[policy.actions[:, 1] < 0, 0]])
                for entry in np.flatnonzero(exact & (policy.actions[:, 1] >= 0)):
                    kind, obj = policy.actions[entry]
                    self.assertTrue(policy.object_actors(encoded, KINDS[kind], int(obj)).any())

    def test_decision_log_probability_is_the_flat_entry_of_kind_and_object(self):
        config, observation, terrain = fixture_commands()
        policy = NeuralPolicy(config, 5)
        for row in rows_for(policy, observation, terrain):
            if row['target']['kind'] not in ('build', 'train', *COMMAND_KINDS):
                continue
            with no_grad():
                trunk = policy._trunk(row['encoded'])
                logits = policy._action_logits(trunk).data
            mask = policy.action_mask(row['encoded'])
            from generals_policy import LOGIT_CAP
            legal = np.flatnonzero(mask)
            capped = np.tanh((logits-logits[legal].max())/LOGIT_CAP)*LOGIT_CAP
            expected = capped[policy.action_choice(row['target'])]-np.log(np.exp(capped[legal]).sum())
            with no_grad():
                _, trace, *_ = policy.decide(row['encoded'], teacher={**row['target']})
            self.assertEqual((trace['kind'], trace.get('template', trace.get('command'))),
                             (row['target']['kind'], action_object(row['target'])))
            # The learner's decision term is that one flat entry's capped log-softmax.
            model = TorchPolicyV3(policy, dtype=torch.float64)
            output = model.teacher_heads([row['encoded']], [row['target']], *model.events([row['encoded']]))
            self.assertAlmostEqual(float(output['by_head']['decision'][0]), float(expected), places=9)


class LSTMTests(unittest.TestCase):
    def setUp(self):
        config, observation, terrain = fixture_commands()
        self.policy = NeuralPolicy(config, 5)
        self.model = TorchPolicyV3(self.policy, dtype=torch.float64)
        rows = rows_for(self.policy, perceived(copy.deepcopy(observation)), terrain)
        self.rows = [rows[i % len(rows)] for i in range(64)]

    def test_stepping_one_decision_at_a_time_equals_one_64_step_unroll(self):
        memory, expected = None, []
        for row in self.rows:
            logp, memory = numpy_logp(self.policy, row, memory)
            expected.append(logp)
        model, encoded = self.model, [row['encoded'] for row in self.rows]
        values, numeric = encode_values(model, encoded)
        plan = teacher_plan(model, encoded, [row['target'] for row in self.rows], numeric.shape[1])
        plan['staged'] = staged = model.stager.stage(plan['values'])
        bundle = model.encode_inputs(model.stager.stage(values), (staged['command_rows'], staged['selected_indices'],
                                                                 staged['selected_lengths']))
        resets = torch.zeros((1, 64), dtype=torch.bool); resets[0, 0] = True
        core, state = model.core(bundle['core_input'][None], resets, model.initial_state(1))
        output = teacher_heads(model, core[0], bundle, plan)
        np.testing.assert_allclose(output['logp'].detach().numpy(), expected, rtol=1e-8, atol=1e-8)
        for (h, c), (h2, c2) in zip(memory, state):
            np.testing.assert_allclose(h.data, h2[0].detach().numpy(), atol=1e-10)
            np.testing.assert_allclose(c.data, c2[0].detach().numpy(), atol=1e-10)

    def test_state_carried_across_two_unrolls_equals_one_pass_and_resets_start_a_new_episode(self):
        model = self.model
        size = model.weights['core_in_w'].shape[0]
        inputs = torch.randn((3, 128, size), dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        resets = torch.zeros((3, 128), dtype=torch.bool)
        resets[:, 0] = True
        resets[1, 70] = True  # lane 1 starts a new episode at step 70
        whole, final = model.core(inputs, resets, model.initial_state(3))
        first, middle = model.core(inputs[:, :64], resets[:, :64], model.initial_state(3))
        second, carried = model.core(inputs[:, 64:], resets[:, 64:], middle)
        torch.testing.assert_close(torch.cat([first, second], dim=1), whole, rtol=0, atol=1e-12)
        for (h, c), (h2, c2) in zip(final, carried):
            torch.testing.assert_close(h, h2, rtol=0, atol=1e-12)
        fresh, _ = model.core(inputs[1:2, 70:], resets[1:2, 70:], model.initial_state(1))
        torch.testing.assert_close(whole[1:2, 70:], fresh, rtol=0, atol=1e-12)
        self.assertFalse(torch.allclose(whole[0, 70:], model.core(inputs[0:1, 70:], resets[0:1, 70:]|True,
                                                                  model.initial_state(1))[0][0]))

    def test_live_policy_carries_the_same_state_as_the_reference(self):
        from torch_live_policy import TorchLivePolicy
        live = TorchLivePolicy(self.policy, device='cpu', dtype=torch.float64)
        left, right = np.random.default_rng(9), np.random.default_rng(9)
        memory = state = None
        for row in self.rows[:20]:
            with no_grad():
                a = self.policy.decide(row['encoded'], memory, rng=left)
            b = live.decide(row['encoded'], state, rng=right)
            self.assertEqual(a[1], b[1])
            self.assertAlmostEqual(float(a[4].data), float(b[4].data), places=6)
            memory, state = a[2], b[2]


class LaneSamplerTests(unittest.TestCase):
    def setUp(self):
        from test_event_policy import EventTests
        fixture_case = EventTests('setUp'); fixture_case.setUp()
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        entries = []
        for number, (length, split) in enumerate(((3, 'train'), (9, 'train'), (5, 'train'), (12, 'train'),
                                                  (4, 'validation'), (7, 'validation'))):
            rows = [raw_row(0, fixture_case.obs)]+[raw_row(30*(i+1), fixture_case.obs, [{'kind': 'stop', 'units': [10]}])
                                                   for i in range(length)]
            game = {'rows': rows, 'map': fixture_case.terrain, 'source_ticks_per_second': 60, 'player': 2, 'replay_sha256': RECORDING}
            result = write_shard(root/f'shards/{number}.zip', [game], fixture_case.policy, {'match_key': f'test:{number}'})
            entries.append({**result, 'path': f'shards/{number}.zip', 'match_key': f'test:{number}', 'split': split})
        _write_json(root/'dataset.json', {'kind': 'generals_event_dataset', 'schema': 1, 'config': fixture_case.config,
                                          'entries': entries})
        from event_dataset import EventDataset
        self.dataset = EventDataset(root/'dataset.json')

    def tearDown(self):
        self.temporary.cleanup()

    def test_every_decision_once_per_pass_in_episode_order(self):
        dataset = self.dataset
        episodes = dataset.episodes('train')
        for epoch in (0, 1):
            seen = {number: [] for number in range(len(episodes))}
            for schedule in dataset.lane_schedule('train', seed=1, epoch=epoch, lanes=3, steps=4, shuffle=False):
                episode, row, reset = schedule
                np.testing.assert_array_equal(reset, (episode >= 0) & (row == 0))
                for lane in range(3):
                    for step in range(4):
                        if episode[lane, step] >= 0:
                            seen[int(episode[lane, step])].append(int(row[lane, step]))
                rows = dataset.lane_rows('train', schedule, range(3))
                self.assertEqual(len(rows), int((episode >= 0).sum()))
                for value in rows:
                    self.assertEqual(int(episode[value['lane'], value['step']]) >= 0, True)
            for number, (_, count) in enumerate(episodes):
                self.assertEqual(seen[number], list(range(count)))
        # Rows reach a lane in game order.
        frames = {}
        for schedule in dataset.lane_schedule('train', seed=1, epoch=0, lanes=2, steps=3, shuffle=False):
            for value in dataset.lane_rows('train', schedule, range(2)):
                frames.setdefault((value['match_key'], value['player']), []).append(value['frame'])
        for values in frames.values():
            self.assertEqual(values, sorted(values))
        self.assertEqual(sum(map(len, frames.values())), sum(count for _, count in episodes))

    def test_a_training_epoch_ends_at_its_last_full_update(self):
        rows = sum(count for _, count in self.dataset.episodes('train'))
        updates = [episode for episode, _, _ in self.dataset.lane_schedule('train', seed=1, epoch=0, lanes=2, steps=3)]
        self.assertGreater(len(updates), 1)
        self.assertTrue(all((episode >= 0).all() for episode in updates))
        full = [episode for episode, _, _ in self.dataset.lane_schedule('train', seed=1, epoch=0, lanes=2, steps=3, shuffle=False)]
        self.assertEqual(sum(int((episode >= 0).sum()) for episode in full), rows)
        # Too few rows for one full update: one partial update, not none.
        (episode, _, _), = self.dataset.lane_schedule('train', seed=1, epoch=0, lanes=1000, steps=3)
        self.assertLess(int((episode >= 0).sum()), 3000)

    def test_validation_matches_never_appear_in_training(self):
        dataset = self.dataset
        train = {dataset.entries[chunks[0][0]]['match_key'] for chunks, _ in dataset.episodes('train')}
        validation = {dataset.entries[chunks[0][0]]['match_key'] for chunks, _ in dataset.episodes('validation')}
        self.assertEqual(validation, {'test:4', 'test:5'})
        self.assertFalse(train & validation)
        for schedule in dataset.lane_schedule('train', seed=2, epoch=0, lanes=2, steps=5):
            self.assertFalse({value['match_key'] for value in dataset.lane_rows('train', schedule, range(2))} & validation)


class GradientAndCheckpointTests(unittest.TestCase):
    def test_every_parameter_learns_and_checkpoints_round_trip(self):
        name, config, observation, terrain = observations()[2]
        # Choices that carry gradient: a charging button, two queued items (two legal slots).
        observation['own_entities'][0]['command_cooldowns'] = [{'command': 'Command_ArtilleryBarrage', 'ready_percent': 40.0}]
        observation['own_entities'][1]['production_queue'].append(
            {'production_id': 4, 'kind': 1, 'progress': 0.0, 'remaining_quantity': 1, 'unit_type': 'ChinaVehicleDozer'})
        # Two scatter channels can both be dead at a unit's pixel after the 1x1 projection; eight are not.
        config = {**config, 'widths': {**config['widths'], 'scatter': 8}}
        policy = NeuralPolicy(config, 5)
        rows = rows_for(policy, observation, terrain)
        learner = EventLearner(policy)
        # One lane through every row in order, so the recurrent weights see a carried state.
        from training_preparation import prepare_update
        update = prepare_update(learner.model, [{**row, 'lane': 0, 'step': step} for step, row in enumerate(rows)],
                                learner.pair_budget)
        resets = np.zeros((1, len(rows)), dtype=bool); resets[0, 0] = True
        update.update(lanes=1, steps=len(rows), resets=resets, present=np.ones((1, len(rows)), dtype=bool))
        learner.run(update, train=True)
        # The value head is untrained in imitation (RL replaces it). The unit inputs' task relations are
        # the memory-free policy's, unchanged, and need task fixtures (test_unit_tasks).
        # The last upsampling bias adds one constant to every position logit, which the softmax cancels.
        from generals_policy import location_stages
        untrained = {'value_w', 'value_b', f'location_up{len(location_stages(config))-1}_b'}
        missing = [name for name, parameter in learner.model.weights.items()
                   if name not in untrained and not name.startswith('task_')
                   and (parameter.grad is None or not parameter.grad.abs().sum())]
        self.assertEqual(missing, [])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.npz'
            learner.save(path, {})
            restored, optimizer, document = NeuralPolicy.load(path)
            for name, parameter in learner.model.weights.items():
                np.testing.assert_array_equal(restored.parameters[name].data, parameter.detach().numpy().astype(np.float64))
            from policy_checkpoints import is_old
            self.assertFalse(is_old(path))
            rng = np.random.default_rng(1)
            with no_grad():
                a = restored.decide(rows[0]['encoded'], rng=np.random.default_rng(1))
            self.assertEqual(a[1], restored.decide(rows[0]['encoded'], rng=rng)[1])


@unittest.skipUnless(all(path.exists() for path in OLD_CHECKPOINTS), 'old checkpoints are local only')
class OldCheckpointTests(unittest.TestCase):
    def test_v11_1500_and_70m_load_through_the_old_modules_and_decide(self):
        from policy_checkpoints import actor_class, is_old, load_policy, policy_modules
        for path in OLD_CHECKPOINTS:
            with self.subTest(path.parent.name):
                self.assertTrue(is_old(path))
                policy, _, _ = load_policy(path)
                self.assertEqual(type(policy).__module__, 'generals_policy_old')
                live = policy_modules(path)[1](policy, device='cpu')
                self.assertEqual(type(live).__module__, 'torch_live_policy_old')
                _, observation, _ = fixture()
                side = policy.config['map_side']*8
                terrain = {'grid_size': side, 'extent_low': [0, 0, 0], 'extent_high': [1000, 1000, 100],
                           'height': [0]*side*side, 'cliff': [0]*side*side}
                observation['visibility_grid'] = [2]*side*side
                # The feature-review engine's 'ability' task: the 70M reads it as 'other' (unit_tasks_old).
                observation['own_entities'][0]['task']['kind'] = 'ability'
                actor = actor_class(live)(live, 3)
                self.assertEqual(type(actor).__name__, 'DelayActor')
                for frame in (0, 300, 600):
                    actor.next_decision_frame = 0
                    actor.act(copy.deepcopy(observation), terrain, frame)
                    self.assertIn(actor.last['trace']['kind'], KINDS)
                del policy, live, actor


if __name__ == '__main__':
    unittest.main()
