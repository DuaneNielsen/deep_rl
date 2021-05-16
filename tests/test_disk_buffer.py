import rl
import numpy as np
import os
from functools import reduce
import pytest
from env.debug import DummyEnv
import numpy.random

def transition_equal(t1, t2):
    for i, field in enumerate(t1):
        if isinstance(field, dict):
            pass
        elif isinstance(field, np.ndarray):
            assert np.allclose(t1[i], t2[i])
        else:
            assert t1[i] == t2[i]


@pytest.fixture
def filename():
    yield 'test.h5'
    os.remove('test.h5')


def transition(state, next_state, action=1, reward=1.0, done=False):
    return state, action, next_state, reward, done, {}


def equal(t1, t2):
    return reduce(lambda x, y: x and y, [a == b for a, b in zip(t1, t2)])


def test_equal(filename):
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(84, 84), state_dtype=np.float32)
    s1 = buffer.statebuffer.append(np.ones((84, 84), dtype=np.float32))
    s2 = buffer.statebuffer.append(np.ones((84, 84), dtype=np.float32))
    assert equal(transition(s1, s2), transition(s1, s2))
    assert equal(transition(s1, s2), transition(s1, s1)) is False
    buffer.close()


def test_stuff(filename):
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(84, 84), state_dtype=np.float32)
    s1 = np.ones((84, 84), dtype=np.float32) * 3.0
    s1 = buffer.statebuffer.append(s1)
    s2 = buffer.statebuffer.append(np.ones((84, 84), dtype=np.float32))
    tr1 = transition(s1, s2)
    buffer.append(*tr1)
    state, action, next_state, reward, done = buffer[0]
    assert np.allclose(state, s1.state())
    assert np.allclose(next_state, s2.state())
    assert action == tr1[1]
    assert reward == tr1[3]
    assert done == tr1[4]
    buffer.close()


def test_buffer(filename):
    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 0.0, True, {})]
    t2 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(1,), state_dtype=np.float64)

    def policy(state):
        return 0

    for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer):
        buffer.append(s, a, s_p, r, d, i)
        if step + 1 == 3:
            break

    assert len(buffer.episodes) == 2
    assert len(buffer[buffer.episodes[0]]) == 2
    assert len(buffer[buffer.episodes[1]]) == 1
    assert len(buffer) == 3

    transition = buffer[0]
    expected = 0, 0, 1, 0.0, False
    transition_equal(transition, expected)

    transition = buffer[1]
    expected = 1, 0, 2, 0.0, True
    transition_equal(transition, expected)

    transition = buffer[2]
    expected = 0, 0, 1, 0.0, True
    transition_equal(transition, expected)

    buffer.close()


def test_load_before_trajectory_terminates(filename):
    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([3]), 0.0, False, {}), (np.array([4]), 0.0, True, {})]
    traj = [t1, t2]
    env = DummyEnv(traj)
    def policy(state):
        return 0
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(1,), state_dtype=np.float32)
    g = rl.step(env, policy, buffer)

    assert len(buffer) == 0
    assert len(buffer.episodes) == 0

    """ second step, intermediate step"""
    step, s, a, s_p, r, d, i, m = next(g)
    buffer.append(s, a, s_p, r, d, i)
    assert len(buffer) == 1
    assert len(buffer.episodes) == 0
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)

    """ third step, trajectory ends """
    step, s, a, s_p, r, d, i, m = next(g)
    buffer.append(s, a, s_p, r, d, i)
    assert len(buffer) == 2
    assert len(buffer.episodes) == 1
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)

    """ forth step, 2nd trajectory resets  """
    step, s, a, s_p, r, d, i, m = next(g)
    buffer.append(s, a, s_p, r, d, i)
    assert len(buffer) == 3
    assert len(buffer.episodes) == 2
    expected_transition = 0, 0, 1, 0.0, False
    transition_equal(buffer[0], expected_transition)
    expected_transition = 1, 0, 2, 1.0, True
    transition_equal(buffer[1], expected_transition)
    expected_transition = 3, 0, 4, 0.0, True
    transition_equal(buffer[2], expected_transition)

    buffer.close()


def test_buffer_iterator(filename):
    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(1,), state_dtype=np.int64)

    def policy(state):
        return 0

    for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer):
        buffer.append(s, a, s_p, r, d, i)
        if step + 1 == 3:
            break

    assert len(buffer) == 3
    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))
    buffer.close()

    buffer = rl.OnDiskReplayBuffer.load(filename)
    assert len(buffer) == 3
    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))
    buffer.close()

    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = rl.OnDiskReplayBuffer.load(filename)
    for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer):
        buffer.append(s, a, s_p, r, d, i)
        if step + 1 == 3:
            break

    assert len(buffer) == 6
    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))
    transition_equal(buffer[3], (0, 0, 1, 0.0, False))
    transition_equal(buffer[4], (1, 0, 2, 1.0, True))
    transition_equal(buffer[5], (0, 0, 1, 0.0, True))
    buffer.close()


def test_trajectory_iterator(filename):
    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(1,), state_dtype=np.int64)

    def policy(state):
        return 0

    for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer):
        buffer.append(s, a, s_p, r, d, i)
        if step + 1 == 3:
            break

    trajectory = buffer[buffer.episodes[0]]
    transition_equal(trajectory[0], (0, 0, 1, 0.0, False, {}))
    transition_equal(trajectory[1], (1, 0, 2, 1.0, True, {}))

    trajectory = buffer[buffer.episodes[1]]
    transition_equal(trajectory[0], (0, 0, 1, 0.0, True, {}))

    buffer.close()


def test_load_child_class(filename):
    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(1,), state_dtype=np.int64)

    def policy(state):
        return 0

    for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer):
        buffer.append(s, a, s_p, r, d, i)
        if step + 1 == 3:
            break

    assert len(buffer) == 3
    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))
    buffer.close()

    class ChildOnDiskReplayBuffer(rl.OnDiskReplayBuffer):
        def __init__(self):
            super().__init__()

    buffer = ChildOnDiskReplayBuffer.load(filename)
    assert len(buffer) == 3
    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))
    buffer.close()

    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = ChildOnDiskReplayBuffer.load(filename)
    for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer):
        buffer.append(s, a, s_p, r, d, i)
        if step + 1 == 3:
            break

    assert len(buffer) == 6
    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (0, 0, 1, 0.0, True))
    transition_equal(buffer[3], (0, 0, 1, 0.0, False))
    transition_equal(buffer[4], (1, 0, 2, 1.0, True))
    transition_equal(buffer[5], (0, 0, 1, 0.0, True))
    buffer.close()


def test_random_split(filename):
    np.random.seed(0)

    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([3]), 0.0, False, {}), (np.array([4]), 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    buffer = rl.OnDiskReplayBuffer.create(filename, state_shape=(1,), state_dtype=np.int64)

    def policy(state):
        return 0

    for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer):
        buffer.append(s, a, s_p, r, d, i)
        if step + 1 == 3:
            break

    assert len(buffer) == 3
    transition_equal(buffer[0], (0, 0, 1, 0.0, False))
    transition_equal(buffer[1], (1, 0, 2, 1.0, True))
    transition_equal(buffer[2], (3, 0, 4, 0.0, True))
    buffer.close()

    train, val = rl.OnDiskReplayBuffer.load_random_splits(filename, [2, 1])

    assert len(train) == 2
    assert len(val) == 1
    transition_equal(train[0], (3, 0, 4, 0.0, True))
    transition_equal(train[1], (1, 0, 2, 1.0, True))
    transition_equal(val[0], (0, 0, 1, 0.0, False))

    train.close()
    val.close()