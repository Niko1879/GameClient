"""The deployed actor of the memory-free (_old) checkpoints: it sleeps for the delay the model chose.

Kept with generals_policy_old until AlphaGenerals' adoption (notes/plan.md Phase 7)."""
from __future__ import annotations

from event_policy import EventActor
from nn_numpy import no_grad


class DelayActor(EventActor):
    """EventActor scheduled by the policy's delay head; no memory, looks, or APM input."""
    def __init__(self, *args, **kwargs):
        # Fixed pace (configure_pace): ticks between decisions and wait masked; None uses the model's own delays and waits.
        self.pace_ticks = None
        super().__init__(*args, **kwargs)

    def configure_pace(self, apm=None):
        """Issue one command every 60/apm game seconds, never waiting (None: the model's own waits and delays).

        The delay is still sampled (later arguments condition on it), it just does not schedule."""
        if apm is not None and not 0 < apm <= 3600:
            raise ValueError('APM must be in (0, 3600]')
        self.pace_ticks = None if apm is None else round(3600/apm)

    def _look(self, bought, observation, terrain, frame):
        encoded = self._encode(observation, terrain, frame)
        with no_grad():
            actions, trace, self.memory, value, logp, entropy = self.policy.decide(
                encoded, self.memory, rng=self.rng, deterministic=self.deterministic,
                **({} if self.pace_ticks is None else {'allow_wait': False}))
        self.previous_frame = frame
        self.next_decision_frame = frame + (self.policy.config['delay_ticks'][trace['delay']]
                                            if self.pace_ticks is None else self.pace_ticks)
        self._decided(encoded, trace, frame, actions, value, logp, entropy)
        return bought + actions
