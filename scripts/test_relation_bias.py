"""The faster coefficient reduction must preserve the actual gradient equation."""
import unittest

import torch

from torch_policy_v3 import _RelationBias


class RelationBiasTests(unittest.TestCase):
    def test_double_gradient_check_including_geometry_gradient(self):
        torch.manual_seed(4)
        relative = torch.randn(2, 3, 3, 4, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(_RelationBias.apply, (relative, weight)))

    def test_constant_geometry_and_noncontiguous_attention_gradient(self):
        torch.manual_seed(9)
        relative = torch.randn(3, 17, 17, 4, dtype=torch.float64)
        left = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        gradient = torch.randn(3, 2, 17, 17, dtype=torch.float64).permute(0, 2, 3, 1)
        expected = relative @ left
        actual = _RelationBias.apply(relative, right)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        expected.backward(gradient)
        actual.backward(gradient)
        torch.testing.assert_close(right.grad, left.grad, atol=1e-11, rtol=1e-11)

    def test_imitation_only_heads_preserve_every_loss_and_parameter_gradient(self):
        from generals_policy import NeuralPolicy
        from test_fixtures import fixture
        from test_policy import examples
        from train_events import EventLearner, event_losses
        config, observation, terrain = fixture()
        left, right = (EventLearner(NeuralPolicy(config, 5)) for _ in range(2))
        rows = examples(left.reference, observation, terrain)
        for row in rows:
            row['target']['delay'] = 119
        encoded, teachers = [r['encoded'] for r in rows], [r['target'] for r in rows]
        outputs = []
        for learner, reduced in ((left, False), (right, True)):
            output = learner.model.teacher_heads(encoded, teachers, *learner.model.events(encoded), loss_only=reduced)
            outputs.append(output)
            event_losses(learner.model, rows, output).sum().backward()
        for name in outputs[0]['by_head']:
            torch.testing.assert_close(outputs[1]['by_head'][name], outputs[0]['by_head'][name], atol=0, rtol=0)
        for name, parameter in left.model.weights.items():
            other = right.model.weights[name]
            self.assertEqual(parameter.grad is None, other.grad is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(other.grad, parameter.grad, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
