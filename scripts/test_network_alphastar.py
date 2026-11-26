"""The temporary comparison networks (network_alphastar.py): learn, checkpoint, play. Delete with it."""
from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest

import numpy as np
import torch

from generals_policy import NeuralPolicy
from test_alphagenerals import observations, rows_for
from train_events import EventLearner


def variant_config(network='alphastar', **extra):
    name, config, observation, terrain = observations()[2]
    config = {**config, 'network': network, 'widths': {**config['widths'], 'scatter': 8, 'core': 48}, **extra}
    observation['own_entities'][0]['command_cooldowns'] = [{'command': 'Command_ArtilleryBarrage', 'ready_percent': 40.0}]
    observation['own_entities'][1]['production_queue'].append(
        {'production_id': 4, 'kind': 1, 'progress': 0.0, 'remaining_quantity': 1, 'unit_type': 'ChinaVehicleDozer'})
    return config, observation, terrain


class AlphaStarNetworkTests(unittest.TestCase):
    network = 'alphastar'
    extra = {}
    # Parameters behind a zero-started output layer: no gradient on the first update.
    first_step_silent = set()

    def setUp(self):
        self.config, observation, terrain = variant_config(self.network, **self.extra)
        self.policy = NeuralPolicy(self.config, 5)
        self.rows = self.prepared(rows_for(self.policy, observation, terrain))

    def prepared(self, rows):
        return rows

    def test_parameters_replace_the_lstm_and_linear_action_layer(self):
        names = set(self.policy.parameters)
        self.assertFalse({name for name in names if name.startswith('core_lstm')})
        self.assertIn('core_hidden_w', names)
        self.assertIn('verb_res15_b', names)
        self.assertEqual(self.policy.parameters['action_w'].data.shape[0], 256)
        self.assertIn('location_skip3', names)
        with self.assertRaises(ValueError):
            NeuralPolicy({**self.config, 'network': 'other'})

    def test_every_parameter_learns_and_checkpoints_round_trip_and_play(self):
        learner = EventLearner(self.policy)
        from training_preparation import prepare_update
        rows = self.rows
        update = prepare_update(learner.model, [{**row, 'lane': 0, 'step': step} for step, row in enumerate(rows)],
                                learner.pair_budget)
        resets = np.zeros((1, len(rows)), dtype=bool); resets[0, 0] = True
        update.update(lanes=1, steps=len(rows), resets=resets, present=np.ones((1, len(rows)), dtype=bool))
        learner.run(update, train=True)
        from generals_policy import location_stages
        untrained = {'value_w', 'value_b', f'location_up{len(location_stages(self.config))-1}_b'} | self.first_step_silent
        missing = [name for name, parameter in learner.model.weights.items()
                   if name not in untrained and not name.startswith('task_')
                   and (parameter.grad is None or not parameter.grad.abs().sum())]
        self.assertEqual(missing, [])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.npz'
            learner.save(path, {})
            from policy_checkpoints import actor_class, is_old, live_policy, load_policy
            self.assertFalse(is_old(path))
            restored, _, _ = load_policy(path)
            for name, parameter in learner.model.weights.items():
                np.testing.assert_array_equal(restored.parameters[name].data, parameter.detach().numpy().astype(np.float64))
            # No NumPy equations: it plays through the PyTorch live policy, memory-free.
            with self.assertRaises(ValueError):
                restored.decide(rows[0]['encoded'])
            live = live_policy(restored, device='cpu', dtype=torch.float64)
            self.assertIs(actor_class(live).__name__, 'EventActor')
            fresh = live.decide(rows[3]['encoded'], None, rng=np.random.default_rng(2))
            carried = live.decide(rows[3]['encoded'], live.decide(rows[0]['encoded'], None, rng=np.random.default_rng(1))[2],
                                  rng=np.random.default_rng(2))
            self.assertEqual(fresh[1], carried[1])
            self.assertAlmostEqual(float(fresh[4].data), float(carried[4].data), places=9)


class MemoryFreeNetworkTests(AlphaStarNetworkTests):
    network = 'memoryfree'

    def test_parameters_replace_the_lstm_and_linear_action_layer(self):
        names = set(self.policy.parameters)
        self.assertFalse({name for name in names if name.startswith(('core_lstm', 'verb_', 'map_skip', 'location_skip'))})
        self.assertIn('core_hidden_w', names)
        self.assertEqual(self.policy.parameters['action_w'].data.shape[0], self.config['widths']['core'])


class PacedNetworkTests(AlphaStarNetworkTests):
    """network='paced': memory-free orders without a trained no-op; pace (from the delay label) sets the live rate."""
    network = 'paced'
    first_step_silent = {'pace_hidden_w', 'pace_hidden_b'}

    def prepared(self, rows):
        # A delay on every row (event_dataset sets it on order rows), some censored, and a known own pace.
        return [{**row, 'apm': 50., 'delay': (30*(step % 4), step % 3 > 0)} for step, row in enumerate(rows)]

    def test_parameters_replace_the_lstm_and_linear_action_layer(self):
        names = set(self.policy.parameters)
        self.assertFalse({name for name in names if name.startswith(('core_lstm', 'verb_', 'map_skip', 'kind_', 'strategy_'))})
        self.assertTrue({'action_w', 'pace_w', 'pace_hidden_w'} <= names)
        self.assertEqual(self.policy.parameters['action_w'].data.shape[0], self.config['widths']['core'])

    def test_decision_is_scored_over_orders_only(self):
        import torch_policy_v3
        from torch_training_inputs import paced_decision
        from training_preparation import prepare_update
        learner = EventLearner(self.policy)
        part = prepare_update(learner.model, self.rows, learner.pair_budget)['microbatches'][0]
        plan, model = part['plan'], learner.model
        inputs = model.stager.stage(plan['values'])
        core = torch.randn((plan['batch'], self.config['widths']['core']), dtype=model.dtype)
        logits = model.action_logits(core)
        timing, _, decision, wait = paced_decision(model, core, logits, inputs, plan, True)
        # A command row's log p is over the orders: no-op's mass never enters it.
        mask = inputs['decision_mask'] & (model.action_kinds != 0)[None]
        row = int(plan['command_rows'][0])
        from torch_training_inputs import cap_logits
        expected = cap_logits(logits, mask).masked_fill(~mask, -torch.inf).log_softmax(dim=1)[row, inputs['decision_choices'][row]]
        self.assertAlmostEqual(float(timing[row].detach()), float(expected.detach()), places=5)
        self.assertTrue((model.action_kinds[decision] != 0).all())

    def test_pace_sets_the_live_order_rate(self):
        from generals_policy import order_rate
        from policy_checkpoints import live_policy
        live = live_policy(self.policy, device='cpu', dtype=torch.float64)
        with torch.no_grad():
            live.model.weights['pace_b'].fill_(math.log(2))
        encoded = self.rows[3]['encoded']
        _, trace, *_ = live.decide(encoded, None, rng=np.random.default_rng(1), snapshots=.01)
        # orders/s = own rate x exp(pace); a look acts with that x its time (snapshots / snapshot rate).
        expected = order_rate(encoded)*2*.01/live.snapshot_rate
        self.assertAlmostEqual(trace['act_probability'], round(expected, 6), places=6)

    def test_delay_targets_run_to_the_next_order_capped(self):
        from event_dataset import DELAY_CAP, delay_targets
        kinds = [('wait', 30), ('move', 100), ('wait', 200), ('train', 250), ('move', 900), ('move', 900), ('move', 1000)]
        rows = [{'player': 1, 'frame': frame, 'target': {'kind': kind}} for kind, frame in kinds]
        delay_targets(rows)
        # move@100 -> 250; train@250 -> 900 is over the cap; same-tick 900 -> 900; the chunk's last order defaulted.
        self.assertEqual([row.get('delay') for row in rows],
                         [None, (150, True), None, (DELAY_CAP, False), (0, True), (100, True), (DELAY_CAP, False)])

    def test_pace_plan_and_exponential_likelihood(self):
        import torch_policy_v3
        from torch_training_inputs import pace_loglikelihood
        from training_preparation import pace_plan
        plan = {'values': {}, 'categories': {}}
        rows = [{'apm': 60., 'delay': (120, True)}, {'apm': 60., 'delay': (360, False)}, {'apm': None, 'delay': (60, True)},
                {'apm': 60., 'delay': (0, True)}]
        pace_plan(plan, rows)
        values = {k: v[0] for k, v in plan['values'].items()}
        np.testing.assert_array_equal(values['pace_rows'], [0, 1, 3])  # No own pace: no pace row.
        np.testing.assert_allclose(values['pace_delay'], [2, 6, 1/60])  # Same tick: one tick.
        np.testing.assert_array_equal(values['pace_observed'], [1, 0, 1])
        model = torch_policy_v3.torch_policy(self.policy, device='cpu', dtype=torch.float64)
        inputs = {k: torch.as_tensor(v) for k, v in values.items()}
        core = torch.zeros((4, self.config['widths']['core']), dtype=torch.float64)
        loglik = pace_loglikelihood(model, core, inputs, 4).detach().numpy()
        # Zero pace: 1 order/s. Observed 2 s: log 1 - 2; censored at 6 s: -6; no pace row: 0.
        np.testing.assert_allclose(loglik, [-2, -6, 0, -1/60], atol=1e-6)

    def test_validation_reports_the_pace_implied_no_op_share(self):
        from training_preparation import prepare_update
        learner = EventLearner(self.policy)
        update = prepare_update(learner.model, self.rows, learner.pair_budget)
        totals = learner.run(update, train=False)
        # Zero pace: orders at the own rate (50/min) against snapshots at the policy's rate, on every row.
        share = self.policy.snapshot_rate/(self.policy.snapshot_rate+50/60)
        self.assertAlmostEqual(totals['expected_defers'], share*len(self.rows), places=4)

    def test_pace_never_shapes_the_core(self):
        import torch_policy_v3
        model = torch_policy_v3.torch_policy(self.policy, device='cpu', dtype=torch.float64)
        core = torch.randn((2, self.config['widths']['core']), dtype=torch.float64, requires_grad=True)
        model.pace(core).sum().backward()
        self.assertIsNone(core.grad)
        self.assertIsNotNone(model.weights['pace_b'].grad)


class MovementNetworkTests(AlphaStarNetworkTests):
    """network='movement': the location head sees the selected units and every entity relative to them."""
    network = 'movement'

    def test_parameters_replace_the_lstm_and_linear_action_layer(self):
        names = set(self.policy.parameters)
        self.assertFalse({name for name in names if name.startswith(('core_lstm', 'verb_', 'map_skip', 'kind_w'))})
        self.assertTrue({'relevance_q_w', 'relevance_k_w', 'location_extra', 'cell_qkv_w', 'cell_mlp_b_w', 'action_w'} <= names)

    def test_location_marks_the_selected_units_and_spreads_relevance_over_entities(self):
        import torch_policy_v3
        from network_alphastar import RELEVANCE_HEADS, entity_cells
        model = torch_policy_v3.torch_policy(self.policy, device='cpu', dtype=torch.float64)
        n = self.config['map_side']
        side = n*8//2
        pixels = torch.tensor([[0, side*side-1, 5*side+9, 0]])  # cells 0, n*n-1, (1, 2); the last is padding
        real = torch.tensor([[1., 1., 1., 0.]])
        cells = entity_cells(self.config, pixels)
        self.assertEqual(cells.tolist(), [[0, n*n-1, n+2, 0]])
        keys = torch.randn((1, 4, model.widths['key']), dtype=torch.float64)
        query = torch.randn((1, 4*n*n), dtype=torch.float64)
        context = {'cells': cells, 'real': real, 'keys': keys, 'selected': torch.tensor([[1, 2]]), 'lengths': torch.tensor([1])}
        with torch.no_grad():
            planes = model.movement_planes(query, context).reshape(1, 1+RELEVANCE_HEADS, n*n)
        # Only the first selected unit counts (length 1): log(1+1) at its cell, nothing elsewhere.
        self.assertAlmostEqual(float(planes[0, 0, n*n-1]), math.log(2))
        self.assertAlmostEqual(float(planes[0, 0].sum()), math.log(2))
        # Each relevance head spreads one unit of attention over the real entities only.
        np.testing.assert_allclose(planes[0, 1:].sum(dim=1).numpy(), 1, rtol=1e-9)
        self.assertTrue((planes[0, 1:, [0, n*n-1, n+2]].sum(dim=1) > .999).all())

    def test_live_location_reads_the_selection(self):
        from policy_checkpoints import live_policy
        from generals_policy import KINDS
        live = live_policy(self.policy, device='cpu', dtype=torch.float64)
        encoded = self.rows[3]['encoded']
        with torch.no_grad():
            trunk = live._trunk(encoded)
            move = live.action_index[(KINDS.index('move'), -1)]
            n, fine = self.config['map_side'], self.config['position_bins']//self.config['map_side']
            trunk['selected'] = [0]
            one = live._location(trunk, move, live._selection_summary(trunk, [0]))
            trunk['selected'] = [1]
            other = live._location(trunk, move, live._selection_summary(trunk, [1]))
        self.assertEqual(tuple(one.shape), (n*n, fine*fine))
        self.assertFalse(torch.equal(one, other))


class EffectsMapNetworkTests(AlphaStarNetworkTests):
    """effects='map' on memory-free: no effect tokens; effects painted on the map; units read their cell."""
    network = 'memoryfree'
    extra = {'effects': 'map'}

    def test_parameters_replace_the_lstm_and_linear_action_layer(self):
        names = set(self.policy.parameters)
        # No effect tokens in the transformer; their pool into the core stays (heard launches without a position).
        self.assertFalse({name for name in names if 'effects' in name and not name.startswith('pool_effects')})
        self.assertTrue({'effect_w', 'effect_scatter_w', 'effect_map_w', 'unit_cell_w', 'unit_cell_key_w',
                         'pool_effects_query'} <= names)
        plain = NeuralPolicy(variant_config('memoryfree')[0], 5)
        self.assertTrue({name for name in plain.parameters if name.startswith('attn0_effects')})
        self.assertEqual(self.policy.parameters['core_in_w'].data.shape, plain.parameters['core_in_w'].data.shape)
        with self.assertRaises(ValueError):
            NeuralPolicy({**self.config, 'network': 'alphastar'})

    def test_feature_key_is_unchanged(self):
        from event_store import feature_identity
        plain = NeuralPolicy(variant_config('memoryfree')[0], parameters=False)
        self.assertEqual(feature_identity(self.policy), feature_identity(plain))

    def test_effects_are_painted_at_their_pixel_and_units_read_their_cell(self):
        import torch_policy_v3
        from generals_policy import MAP_POOL, SCATTER_SCALE
        from training_preparation import encode_values
        model = torch_policy_v3.torch_policy(self.policy, device='cpu', dtype=torch.float64)
        encoded = self.rows[0]['encoded']
        self.assertTrue(len(encoded['effect_numeric']))
        values, _ = encode_values(model, [encoded])
        inputs = model.stager.stage(values)
        side = self.config['map_side']*SCATTER_SCALE//MAP_POOL
        with torch.no_grad():
            tokens = model.entity_inputs(inputs)[1]
            image = model.effect_image(inputs, tokens)
            self.assertEqual(tuple(image.shape), (1, self.config['widths']['scatter'], side, side))
            painted = set(torch.nonzero(image[0].abs().sum(0).flatten()).flatten().tolist())
            placed = inputs['effect_placed'][0] > 0
            self.assertTrue(placed.any())
            self.assertLessEqual(painted, set(inputs['effect_pixels'][0][placed].tolist()))
            # Effects without a position are never painted.
            unplaced = inputs.copy(); unplaced['effect_placed'] = torch.zeros_like(inputs['effect_placed'])
            self.assertFalse(model.effect_image(unplaced, tokens).abs().sum())
            # Each real unit gets what the map holds at its cell; padding stays zero.
            out = model.encode_inputs(inputs)
            features, keys = model.cell_context(inputs, out['map'], torch.zeros_like(out['features']), torch.zeros_like(out['keys']))
            from network_alphastar import entity_cells
            real = inputs['real'][0] > 0
            cells = entity_cells(self.config, inputs['scatter_pixels'])[0]
            here = out['map'][0].flatten(1).T[cells]
            p = model.weights
            np.testing.assert_allclose(features[0, real].numpy(), (here @ p['unit_cell_w']+p['unit_cell_b'])[real].numpy(), atol=1e-12)
            np.testing.assert_allclose(keys[0, real].numpy(), (here @ p['unit_cell_key_w'])[real].numpy(), atol=1e-12)
            self.assertTrue(features[0, real].abs().sum() > 0)
            self.assertFalse(features[0, ~real].abs().sum())
            # A row without effects: nothing painted, an empty pool.
            plain = dict(encoded, effect_numeric=encoded['effect_numeric'][:0], effect_types=encoded['effect_types'][:0],
                         effect_powers=encoded['effect_powers'][:0])
            bare = model.stager.stage(encode_values(model, [plain])[0])
            self.assertTrue(torch.isfinite(model.encode_inputs(bare)['core_input']).all())
            # Moving every effect elsewhere changes what the units read.
            moved = inputs.copy(); moved['effect_pixels'] = (inputs['effect_pixels']+side*side//2) % (side*side)
            self.assertFalse(torch.allclose(model.encode_inputs(moved)['features'], out['features']))


if __name__ == '__main__':
    unittest.main()
