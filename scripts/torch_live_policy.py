"""PyTorch inference with the reference policy's autoregressive masks and sampling."""
from __future__ import annotations

from copy import deepcopy

import torch

from generals_policy import NeuralPolicy
from nn_numpy import Tensor
from torch_policy_v3 import TorchPolicyV3


class TorchLivePolicy(NeuralPolicy):
    def __init__(self, reference, *, device='cuda', dtype=torch.float32):
        if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('GPU inference requires a PyTorch installation with a working ROCm or CUDA device')
        super().__init__(reference.config, parameters=False)
        self.model = TorchPolicyV3(reference, device=device, dtype=dtype).eval()
        # The live actor applies the checkpoint's rating switch when it prepares the row.
        self.model.config = {**self.config, 'rating_input': True}
        self.parameters = self.model.weights
        self._sample_array = self.model.array
        self.rating_input = getattr(reference, 'rating_input', False)

    _sample_concat = staticmethod(torch.cat)
    _sample_normalize = staticmethod(TorchPolicyV3.normalize)

    def _sampling_logits(self, logits):
        # Transfer just the current head. Sharing the NumPy draw preserves temperature,
        # masks, seeded sampling and the reference probability calculation.
        return Tensor(logits.cpu().numpy())

    def _template_embeddings(self):
        return self.model.object_embedding(self.model.identity_template_objects)

    def _trunk(self, encoded):
        core, entities, spatial = self.model.events([encoded])
        size = self.config['hidden']
        return core[0, :size], entities[0], spatial, self.model.values(core)[0], core[0, size:]

    def _location(self, spatial, context, build_features=None):
        build = None if build_features is None else (
            self.model.indices([0]), self.model.array(build_features[None]))
        cells, logits = self.model.location(spatial, context.unsqueeze(0), build)
        return cells[0], logits[0]

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
            hidden, _, spatial, _, gate = self._trunk(actor.last['encoded'])
            self._location(spatial, self._action_type(hidden, gate)[1])[1].cpu()
        finally:
            vars(actor).clear()
            vars(actor).update(state)
            actor.counters = counters
            actor.rng.bit_generator.state = rng
