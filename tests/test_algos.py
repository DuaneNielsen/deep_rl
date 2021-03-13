import torch
import algos.a2c as a2c


def test_td():
    bootstrap_value = torch.tensor([[1.0]])
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(1)
    done = torch.tensor([False, False, False, False]).unsqueeze(1)
    target_values = a2c.td_targets(bootstrap_value, rewards, done, discount=1.0)

    assert torch.allclose(target_values,
                          torch.tensor([2.0, 2.0, 2.0, 2.0]).unsqueeze(1))

    bootstrap_value = torch.tensor([[1.0]])
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(1)
    done = torch.tensor([False, True, False, False]).unsqueeze(1)
    target_values = a2c.td_targets(bootstrap_value, rewards, done, discount=1.0)

    assert torch.allclose(target_values,
                          torch.tensor([0.0, 0.0, 2.0, 2.0]).unsqueeze(1))

    bootstrap_value = torch.tensor([[1.0]])
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
    done = torch.tensor([False, True, False, False]).unsqueeze(1)
    target_values = a2c.td_targets(bootstrap_value, rewards, done, discount=1.0)

    assert torch.allclose(target_values,
                          torch.tensor([1.0, 1.0, 2.0, 2.0]).unsqueeze(1))

    bootstrap_value = torch.tensor([[1.0]])
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
    done = torch.tensor([False, True, False, False]).unsqueeze(1)
    target_values = a2c.td_targets(bootstrap_value, rewards, done, discount=0.9)

    assert torch.allclose(target_values,
                          torch.tensor([0.9, 1.0, 1.9 * 0.9, 1.9]).unsqueeze(1))