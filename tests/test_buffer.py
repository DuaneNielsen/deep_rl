import buffer as run
import pytest
from env.trivial import DummyEnv
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
    runner = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    runner.attach_observer("replay_buffer", buffer)

    def policy(state):
        return 0

    run.episode(runner, policy)
    run.episode(runner, policy)

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
    env = DummyEnv([])
    runner = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    runner.attach_observer("replay_buffer", buffer)

    """ first step, from env reset """
    step = 0, 0, 0.0, False, {}
    runner.observe_step(*step)
    assert len(buffer) == 0
    assert len(buffer.trajectories) == 0

    """ second step, intermediate step"""
    step = 0, 1, 0.0, False, {}
    runner.observe_step(*step)
    assert len(buffer) == 1
    assert len(buffer.trajectories) == 0
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)

    """ third step, trajectory ends """
    step = 0, 2, 1.0, True, {}
    runner.observe_step(*step)
    assert len(buffer) == 2
    assert len(buffer.trajectories) == 1
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)

    """ forth step, 2nd trajectory resets  """
    step = 0, 3, 0.0, False, {}
    runner.observe_step(*step)
    assert len(buffer) == 2
    assert len(buffer.trajectories) == 1
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)

    """ fifth step, 2nd trajectory ends """
    step = 0, 4, 1.0, True, {}
    runner.observe_step(*step)
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
    env = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    env.attach_observer("replay_buffer", buffer)

    def policy(state):
        return 0

    run.episode(env, policy)
    run.episode(env, policy)

    assert len(buffer) == 3

    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))


def test_trajectory_iterator():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {'s': 1}), (2, 1.0, True, {'s': 2})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {'s': 1})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    env.attach_observer("replay_buffer", buffer)

    def policy(state):
        return 0

    run.episode(env, policy)
    run.episode(env, policy)

    trajectory = run.TrajectoryTransitions(buffer, buffer.trajectories[0])
    transition_equal(next(trajectory), (0, 0, 1, 0.0, False, {'s': 1}))
    transition_equal(next(trajectory), (1, 0, 2, 1.0, True, {'s': 2}))
    with pytest.raises(StopIteration):
        next(trajectory)

    trajectory = run.TrajectoryTransitions(buffer, buffer.trajectories[1])
    transition_equal(next(trajectory), (0, 0, 1, 0.0, True, {'s': 1}))

    with pytest.raises(StopIteration):
        next(trajectory)


def test_reverse_trajectory_iterator():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {'s': 1}), (2, 1.0, True, {'s': 2})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {'s': 1})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    env.attach_observer("replay_buffer", buffer)

    def policy(state):
        return 0

    run.episode(env, policy)
    run.episode(env, policy)

    trajectory = run.TrajectoryTransitionsReverse(buffer, buffer.trajectories[0])
    transition_equal(next(trajectory), (1, 0, 2, 1.0, True, {'s': 2}))
    transition_equal(next(trajectory), (0, 0, 1, 0.0, False, {'s': 1}))
    with pytest.raises(StopIteration):
        next(trajectory)

    trajectory = run.TrajectoryTransitions(buffer, buffer.trajectories[1])
    transition_equal(next(trajectory), (0, 0, 1, 0.0, True, {'s': 1}))

    with pytest.raises(StopIteration):
        next(trajectory)


def test_enrichment_returns():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    env.attach_observer("replay_buffer", buffer)

    buffer.attach_enrichment(run.Returns())

    def policy(state):
        return 0

    run.episode(env, policy)
    run.episode(env, policy)

    dataset = run.ReplayBufferDataset(buffer, info_keys=['g'])

    assert dataset[0].g == 2.0
    assert dataset[1].g == 2.0
    assert dataset[2].g == 1.0

    assert dataset[3].g == 0.0


def test_enrichment_discounted_returns():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    env.attach_observer("replay_buffer", buffer)

    discount = 0.9

    buffer.attach_enrichment(run.DiscountedReturns(discount=discount))

    def policy(state):
        return 0

    run.episode(env, policy)
    run.episode(env, policy)

    dataset = run.ReplayBufferDataset(buffer, info_keys=['g'])

    assert dataset[0].g == discount * 1.0 + discount ** 2 * 1.0
    assert dataset[1].g == 1.0 + discount * 1.0
    assert dataset[2].g == 1.0

    assert dataset[3].g == 0.0


def test_dataload():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (3, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    env = run.SubjectWrapper(env)
    buffer = run.ReplayBuffer()
    env.attach_observer("replay_buffer", buffer)

    buffer.attach_enrichment(run.Returns())

    def policy(state):
        return 0

    run.episode(env, policy)
    run.episode(env, policy)

    dataset = run.ReplayBufferDataset(buffer, info_keys=['g'])
    dl = DataLoader(dataset, batch_size=4)

    for data in dl:
        assert torch.allclose(data.s, torch.tensor([0, 1, 2, 0]))
        assert torch.allclose(data.a, torch.tensor([0, 0, 0, 0]))
        assert torch.allclose(data.r, torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.double))
        assert (~torch.bitwise_xor(data.d, torch.tensor([False, False, True, True]))).all()
        assert torch.allclose(data.s_p, torch.tensor([1, 2, 3, 1]))
        assert torch.allclose(data.g, torch.tensor([2.0, 2.0, 1.0, 0.0], dtype=torch.double))