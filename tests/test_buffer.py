import buffer as bf
import pytest

import driver
from env.debug import DummyEnv


def transition_equal(t1, t2):
    for i, field in enumerate(t1):
        if isinstance(field, dict):
            pass
        else:
            assert t1[i] == t2[i]


def test_buffer():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 0.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = bf.ReplayBuffer()

    def policy(state):
        return 0

    for global_step, transition in enumerate(driver.step_environment(env, policy)):
        buffer.append(*transition)
        if global_step + 1 == 3:
            break

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
    buffer = bf.ReplayBuffer()

    """ first step, from env reset """
    s = env.reset()
    assert len(buffer) == 0
    assert len(buffer.trajectories) == 0

    """ second step, intermediate step"""
    a = 0
    s_p, r, d, i = env.step(a)
    buffer.append(s, a, s_p, r, d, i)
    assert len(buffer) == 1
    assert len(buffer.trajectories) == 0
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)

    """ third step, trajectory ends """
    s = s_p
    a = 0
    s_p, r, d, i = env.step(a)
    buffer.append(s, a, s_p, r, d, i)
    assert len(buffer) == 2
    assert len(buffer.trajectories) == 1
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)

    """ forth step, 2nd trajectory resets  """
    s = env.reset()
    assert len(buffer) == 2
    assert len(buffer.trajectories) == 1
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)

    """ fifth step, 2nd trajectory ends """
    a = 0
    s_p, r, d, i = env.step(a)
    buffer.append(s, a, s_p, r, d, i)
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
    buffer = bf.ReplayBuffer()

    def policy(state):
        return 0

    for global_step, transition in enumerate(driver.step_environment(env, policy)):
        buffer.append(*transition)
        if global_step + 1 == 3:
            break

    assert len(buffer) == 3

    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))


def test_trajectory_iterator():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {'s': 1}), (2, 1.0, True, {'s': 2})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {'s': 1})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = bf.ReplayBuffer()

    def policy(state):
        return 0

    for global_step, transition in enumerate(driver.step_environment(env, policy)):
        buffer.append(*transition)
        if global_step + 1 == 3:
            break

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
    buffer = bf.ReplayBuffer()

    def policy(state):
        return 0

    for global_step, transition in enumerate(driver.step_environment(env, policy)):
        buffer.append(*transition)
        if global_step + 1 == 3:
            break

    trajectory = bf.TrajectoryTransitionsReverse(buffer, buffer.trajectories[0])
    transition_equal(next(trajectory), (1, 0, 2, 1.0, True, {'s': 2}))
    transition_equal(next(trajectory), (0, 0, 1, 0.0, False, {'s': 1}))
    with pytest.raises(StopIteration):
        next(trajectory)

    trajectory = bf.TrajectoryTransitions(buffer, buffer.trajectories[1])
    transition_equal(next(trajectory), (0, 0, 1, 0.0, True, {'s': 1}))

    with pytest.raises(StopIteration):
        next(trajectory)


def test_trajectory_info():
    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = bf.ReplayBuffer()

    def policy(state):
        return 0

    for global_step, transition in enumerate(driver.step_environment(env, policy)):
        buffer.append(*transition)
        if global_step + 1 == 4:
            break

    assert buffer.trajectory_info[0]['return'] == 2.0
    assert buffer.trajectory_info[0]['len'] == 3

    assert buffer.trajectory_info[1]['return'] == 0.0
    assert buffer.trajectory_info[1]['len'] == 1


def test_append_trajectory():

    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 1.0, False, {}), (2, 1.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer1 = bf.ReplayBuffer()

    def policy(state):
        return 0

    for global_step, transition in enumerate(driver.step_environment(env, policy)):
        buffer1.append(*transition)
        if global_step + 1 == 4:
            break

    t1 = [(5, 0.0, False, {}), (6, 0.0, False, {}), (7, 1.0, False, {}), (8, 1.0, True, {})]
    t2 = [(9, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer2 = bf.ReplayBuffer()

    def policy(state):
        return 0

    for global_step, transition in enumerate(driver.step_environment(env, policy)):
        buffer2.append(*transition)
        if global_step + 1 == 4:
            break

    assert len(buffer1) == 4
    assert len(buffer2) == 4

    buffer1.append_buffer(buffer2)

    assert len(buffer1) == 2 * len(buffer2)

    assert buffer1[4].s == 5
    assert buffer1[0].s == 0
    assert buffer1[7].s == 9