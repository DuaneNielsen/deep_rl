import buffer as bf
import pytest

import driver
import observer
from env.debug import DummyEnv
import torch
from torch.utils.data import DataLoader


def transition_equal(t1, t2):
    for i, field in enumerate(t1):
        if isinstance(field, dict):
            assert t1[i]['s'] == t2[i]['s']
        else:
            assert t1[i] == t2[i]


def test_buffer():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 0.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    def policy(state):
        return 0

    driver.episode(buffer, policy)
    driver.episode(buffer, policy)

    start, end = buffer.trajectories[0]
    assert len(buffer.buffer[start:end]) == 3
    start, end = buffer.trajectories[1]
    assert len(buffer.buffer[start:end]) == 2
    assert len(buffer.transitions) == 3

    transition = buffer[0]
    expected = 0, 0, 1, 0.0, False
    transition_equal(transition, expected)

    transition = buffer[1]
    expected = 1, 0, 2, 0.0, True
    transition_equal(transition, expected)

    transition = buffer[2]
    expected = 0, 0, 1, 0.0, True
    transition_equal(transition, expected)


def test_load_before_trajectory_terminates():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, True, {})]
    t2 = [(3, 0.0, False, {}), (4, 1.0, True, {})]
    traj = [t1, t2]
    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    """ first step, from env reset """
    env.reset()
    assert len(buffer) == 0
    assert len(buffer.trajectories) == 0

    """ second step, intermediate step"""
    env.step(0)
    assert len(buffer) == 1
    assert len(buffer.trajectories) == 0
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)

    """ third step, trajectory ends """
    env.step(0)
    assert len(buffer) == 2
    assert len(buffer.trajectories) == 1
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)

    """ forth step, 2nd trajectory resets  """
    env.reset()
    assert len(buffer) == 2
    assert len(buffer.trajectories) == 1
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)

    """ fifth step, 2nd trajectory ends """
    env.step(0)
    assert len(buffer) == 3
    assert len(buffer.trajectories) == 2
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)
    expected_transition = 3, 0, 4, 1.0, True
    transition_equal(buffer[2], expected_transition)


def test_buffer_iterator():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    def policy(state):
        return 0

    driver.episode(env, policy)
    driver.episode(env, policy)

    assert len(buffer) == 3

    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))


def test_trajectory_iterator():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {'s': 1}), (2, 1.0, True, {'s': 2})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {'s': 1})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    def policy(state):
        return 0

    driver.episode(env, policy)
    driver.episode(env, policy)

    trajectory = bf.TrajectoryTransitions(buffer, buffer.trajectories[0])
    transition_equal(next(trajectory), (0, 0, 1, 0.0, False, {'s': 1}))
    transition_equal(next(trajectory), (1, 0, 2, 1.0, True, {'s': 2}))
    with pytest.raises(StopIteration):
        next(trajectory)

    trajectory = bf.TrajectoryTransitions(buffer, buffer.trajectories[1])
    transition_equal(next(trajectory), (0, 0, 1, 0.0, True, {'s': 1}))

    with pytest.raises(StopIteration):
        next(trajectory)


def test_reverse_trajectory_iterator():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {'s': 1}), (2, 1.0, True, {'s': 2})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {'s': 1})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    def policy(state):
        return 0

    driver.episode(env, policy)
    driver.episode(env, policy)

    trajectory = bf.TrajectoryTransitionsReverse(buffer, buffer.trajectories[0])
    transition_equal(next(trajectory), (1, 0, 2, 1.0, True, {'s': 2}))
    transition_equal(next(trajectory), (0, 0, 1, 0.0, False, {'s': 1}))
    with pytest.raises(StopIteration):
        next(trajectory)

    trajectory = bf.TrajectoryTransitions(buffer, buffer.trajectories[1])
    transition_equal(next(trajectory), (0, 0, 1, 0.0, True, {'s': 1}))

    with pytest.raises(StopIteration):
        next(trajectory)


def test_enrichment_returns():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    buffer.enrich(bf.Returns())

    def policy(state):
        return 0

    driver.episode(env, policy)
    driver.episode(env, policy)

    dataset = bf.ReplayBufferDataset(buffer, info_keys=['g'])

    assert dataset[0].g == 2.0
    assert dataset[1].g == 2.0
    assert dataset[2].g == 1.0

    assert dataset[3].g == 0.0


def test_enrichment_discounted_returns():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    discount = 0.9

    buffer.enrich(bf.DiscountedReturns(discount=discount))

    def policy(state):
        return 0

    driver.episode(env, policy)
    driver.episode(env, policy)

    dataset = bf.ReplayBufferDataset(buffer, info_keys=['g'])

    assert dataset[0].g == discount * 1.0 + discount ** 2 * 1.0
    assert dataset[1].g == 1.0 + discount * 1.0
    assert dataset[2].g == 1.0

    assert dataset[3].g == 0.0


def test_dataload():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (3, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    buffer.enrich(bf.Returns())

    def policy(state):
        return 0

    driver.episode(env, policy)
    driver.episode(env, policy)

    dataset = bf.ReplayBufferDataset(buffer, info_keys='g')
    dl = DataLoader(dataset, batch_size=4)

    for data in dl:
        assert torch.allclose(data.s, torch.tensor([0, 1, 2, 0]))
        assert torch.allclose(data.a, torch.tensor([0, 0, 0, 0]))
        assert torch.allclose(data.r, torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.double))
        assert (~torch.bitwise_xor(data.d, torch.tensor([False, False, True, True]))).all()
        assert torch.allclose(data.s_p, torch.tensor([1, 2, 3, 1]))
        assert torch.allclose(data.g, torch.tensor([2.0, 2.0, 1.0, 0.0], dtype=torch.double))

    dataset = bf.ReplayBufferDataset(buffer, fields=('s', 'a'))
    dl = DataLoader(dataset, batch_size=4)

    for data in dl:
        assert torch.allclose(data.s, torch.tensor([0, 1, 2, 0]))
        assert torch.allclose(data.a, torch.tensor([0, 0, 0, 0]))
        assert hasattr(data, 's_p') == False
        assert hasattr(data, 'r') == False
        assert hasattr(data, 'd') == False
        assert hasattr(data, 'g') == False


def test_trajectory_info():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env, buffer = bf.wrap(env)

    def policy(state):
        return 0

    driver.episode(env, policy)
    driver.episode(env, policy)

    assert buffer.trajectory_info[0]['R'] == 2.0
    assert buffer.trajectory_info[0]['len'] == 3

    assert buffer.trajectory_info[1]['R'] == 0.0
    assert buffer.trajectory_info[1]['len'] == 1
