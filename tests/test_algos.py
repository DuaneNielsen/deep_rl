import torch
from algos.dqn import max_q, replicate_states, replicate_actions


def test_replicate_states():

    s1 = [0.0, 1.0, 2.0, 3.0]
    s2 = [1.0, 2.0, 4.0, 4.0]

    state = torch.tensor([
        s1,
        s2
    ])

    states = replicate_states(state, 3)
    assert states.shape[0] == 3 * 2
    assert states.shape[1] == 4
    assert torch.allclose(states[0], torch.tensor(s1))
    assert torch.allclose(states[1], torch.tensor(s1))
    assert torch.allclose(states[2], torch.tensor(s1))
    assert torch.allclose(states[3], torch.tensor(s2))
    assert torch.allclose(states[4], torch.tensor(s2))
    assert torch.allclose(states[5], torch.tensor(s2))

def test_replicate_actions():

    actions = replicate_actions(N=2, n_actions=3)
    assert torch.allclose(actions,
                          torch.tensor([
                              [1.0, 0, 0],
                              [0, 1.0, 0],
                              [0, 0, 1.0],
                              [1.0, 0, 0],
                              [0, 1.0, 0],
                              [0, 0, 1.0]
                          ])
                          )

def test_max_q():

    def q(state, action):
        x = torch.cat([state, action], dim=1)
        return torch.tensor([0, 1.0, 2.0, 0]).unsqueeze(1)

    state = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0, 4.0]
    ])

    value, index = max_q(q, state, 2)

    assert torch.allclose(index, torch.tensor([[1], [0]]))
    assert torch.allclose(value, torch.tensor([[1.0], [2.0]]))
