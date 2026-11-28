"""PyTorch inference with the reference policy's masks, sampling and carried LSTM state."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from generals_policy import NeuralPolicy
from nn_numpy import Tensor
from torch_policy_v3 import torch_policy


class TorchLivePolicy(NeuralPolicy):
    """NeuralPolicy.decide with every network step on PyTorch (batch 1), via TorchPolicyV3's
    batched functions: the training equations, so live play and the learner cannot drift apart."""
    def __init__(self, reference, *, device='cuda', dtype=torch.float32):
        if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('GPU inference requires a PyTorch installation with a working ROCm or CUDA device')
        super().__init__(reference.config, parameters=False)
        self.model = torch_policy(reference, device=device, dtype=dtype).eval()
        # Play never reads the reference's NumPy weights again: keep a weightless one, so a game holds a single
        # (torch) copy of the model (28 Nov: the float64 copy was ~0.75 GB of a deep game's RAM).
        self.model.reference = NeuralPolicy(reference.config, parameters=False)
        self.model.reference.snapshot_rate = reference.snapshot_rate
        # The live actor applies the checkpoint's rating switch when it prepares the row.
        self.model.config = {**self.config, 'rating_input': True}
        self.parameters = self.model.weights
        self.rating_input = getattr(reference, 'rating_input', False)
        self.snapshot_rate = reference.snapshot_rate

    def _sampling_logits(self, logits):
        # Transfer just the current head. Sharing the NumPy draw preserves temperature,
        # masks, seeded sampling and the reference probability calculation.
        return Tensor(logits.float().cpu().numpy())

    def initial_state(self):
        return self.model.initial_state(1)

    def _trunk(self, encoded, state=None):
        model = self.model
        out = model.encode_batch([encoded])
        core, state = model.core(out['core_input'][:, None], torch.zeros((1, 1), dtype=torch.bool, device=model.device),
                                 self.initial_state() if state is None else state)
        return {'core': core[:, 0], 'state': state, 'value': Tensor(float(model.values(core[:, 0])[0])),
                'keys': out['keys'], 'features': out['features'], 'map': out['map'],
                'pixels': out['pixels'], 'real': out['real']}

    def _action(self, action):
        return self.model.indices([action])

    def _pace(self, trunk):
        # constant_pace (evaluate_events --constant-pace): act at the APM input's own rate whatever the state, no-op
        # never drawn: pace switched off for network='paced', a forced constant rate for any other policy.
        if getattr(self, 'constant_pace', False):
            return 0.
        from generals_policy import paced_network
        if not paced_network(self.config):
            return None
        return float(self.model.pace(trunk['core'])[0])

    def _action_logits(self, trunk):
        return self.model.action_logits(trunk['core'])[0]

    def _unit_begin(self, trunk, action):
        model = self.model
        zeros = trunk['core'].new_zeros((1, model.widths['unit_query']))
        return {'context': model.unit_context(trunk['core'], self._action(action)), 'state': (zeros, zeros),
                'action': self._action(action), 'keys': model.unit_keys(trunk['keys'])}

    def _unit_scores(self, trunk, unit):
        model = self.model
        state = model.unit_step(model.unit_query(unit['context']), unit['state'])
        return model.unit_logits(state[0], unit['keys'], unit['action'])[0], {**unit, 'state': state}

    def _unit_add(self, trunk, unit, index):
        return {**unit, 'context': self.model.unit_add(unit['context'], trunk['keys'][:, index])}

    def _selection_summary(self, trunk, selected):
        model = self.model
        return model.selection_summary(trunk['features'], model.indices([0]), model.indices([selected or [0]]),
                                       model.indices([len(selected)]))

    def _slot_logits(self, trunk, action, summary):
        return self.model.slot_logits(trunk['core'], self._action(action), summary)[0]

    def _target_logits(self, trunk, action, summary):
        return self.model.target_logits(trunk['core'], self._action(action), summary, trunk['keys'])[0]

    def _angle_logits(self, trunk, action):
        return self.model.angle_logits(trunk['core'], self._action(action))[0]

    def _location(self, trunk, action, summary, build_features=None):
        build = None if build_features is None else (
            self.model.indices([0]), self.model.array(build_features[None]))
        if self.config.get('network') != 'movement':
            return self.model.location(trunk['map'], trunk['core'], self._action(action), summary, build)[0]
        from network_alphastar import movement_context
        selected = trunk.get('selected') or []
        context = movement_context(self.model, trunk, self.model.indices([0]), trunk['keys'],
                                   self.model.indices([selected or [0]]), self.model.indices([len(selected)]))
        return self.model.location(trunk['map'], trunk['core'], self._action(action), summary, build, context)[0]

    @torch.inference_mode()
    def decide(self, *args, **kwargs):
        return super().decide(*args, **kwargs)

    @torch.inference_mode()
    def warm_up(self, actor, observation, terrain, frame):
        """Run the first observation's GPU work once and discard it, leaving the actor unchanged.

        The first inference (~2 s on the Radeon) and the first map-location call (~0.2 s) compile
        kernels. A headed engine holds the game until the first step, so they cost no game time."""
        state, rng = dict(vars(actor)), actor.rng.bit_generator.state
        counters = actor.counters.copy()
        try:
            actor.next_decision_frame = 0
            actor.act(deepcopy(observation), terrain, frame)
            trunk = self._trunk(actor.last['encoded'])
            self._location(trunk, 0, self._selection_summary(trunk, [])).cpu()
        finally:
            vars(actor).clear()
            vars(actor).update(state)
            actor.counters = counters
            actor.rng.bit_generator.state = rng
