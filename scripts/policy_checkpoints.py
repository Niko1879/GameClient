"""Load a policy checkpoint with the modules that trained it.

AlphaGenerals checkpoints load through generals_policy and torch_live_policy. Until its
adoption (notes/plan.md Phase 7), the memory-free policy's checkpoints (v11-1500, the 70M)
load through the frozen generals_policy_old and torch_live_policy_old, chosen by their
parameter names, so head to head and headed play can run both.
"""
from __future__ import annotations

import numpy as np

# A parameter only the memory-free policy has: its trunk's input layer.
OLD_MARKER = 'p_event_input_w'


def is_old(path):
    """Whether a checkpoint holds the memory-free (_old) policy."""
    with np.load(path, allow_pickle=False) as archive:
        return OLD_MARKER in archive.files


def policy_modules(path):
    """(NeuralPolicy class, TorchLivePolicy class) that load and play this checkpoint."""
    if is_old(path):
        from generals_policy_old import NeuralPolicy
        from torch_live_policy_old import TorchLivePolicy
    else:
        from generals_policy import NeuralPolicy
        from torch_live_policy import TorchLivePolicy
    return NeuralPolicy, TorchLivePolicy


def load_policy(path):
    """(policy, optimizer, document) as NeuralPolicy.load returns them, from the checkpoint's own modules."""
    return policy_modules(path)[0].load(path)


def actor_class(policy):
    """The deployed actor for a loaded or live policy: the memory-free policies schedule by their delay head."""
    if type(policy).__module__ in ('generals_policy_old', 'torch_live_policy_old'):
        from event_policy_old import DelayActor
        return DelayActor
    from event_policy import EventActor
    return EventActor


def live_policy(reference, **options):
    """The PyTorch live policy for a loaded reference policy, from the reference's own generation."""
    if type(reference).__module__ == 'generals_policy_old':
        from torch_live_policy_old import TorchLivePolicy
    else:
        from torch_live_policy import TorchLivePolicy
    return TorchLivePolicy(reference, **options)
