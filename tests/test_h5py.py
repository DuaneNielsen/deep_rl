import pytest
import buffer_h5 as b5
import os
import numpy as np
from env.debug import DummyEnv


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
    if os.path.exists('test.h5'):
        os.remove('test.h5')
    yield 'test.h5'
    os.remove('test.h5')


shape = (210, 160, 3)
dtype = np.uint8

s1 = np.random.randint(shape, dtype=dtype)
a1 = np.random.randint(1, dtype=np.uint8)
r1 = 0.0
d1 = False

s2 = np.random.randint(shape, dtype=dtype)
a2 = np.random.randint(1, dtype=np.uint8)
r2 = 0.0
d2 = False


def pop_buffer(filename):
    t1 = [(np.array([0]), 0.0, False, {}), (np.array([1]), 0.0, False, {}), (np.array([2]), 1.0, True, {})]
    t2 = [(np.array([3]), 0.0, False, {}), (np.array([4]), 0.0, True, {})]
    t3 = [(np.array([5]), 0.0, False, {}), (np.array([6]), 1.0, False, {})]
    traj = [t1, t2, t3]

    env = DummyEnv(traj)
    b = b5.Buffer()
    b.create(filename, state_shape=(1, ), state_dtype=np.int64)

    def policy(state):
        return 0

    for step, s, a, s_p, r, d, i, m in b.step(env, policy, b):
        if step + 1 == 4:
            break

    return b


def assert_row(b, row, s, a, r, d):
    assert np.allclose(b.state[row], np.array([s]))
    assert np.allclose(b.action[row], np.array([a]))
    assert b.reward[row] == r
    assert b.done[row] == d


def test_pop_buffer(filename):
    b = pop_buffer(filename)
    b.close()
    b.load(filename)

    assert b.num_episodes == 3
    assert b.steps == 7

    assert_row(b, 0, 0, 0, 0.0, False)
    assert_row(b, 1, 1, 0, 0.0, False)
    assert_row(b, 2, 2, 0, 1.0, True)
    assert_row(b, 3, 3, 0, 0.0, False)
    assert_row(b, 4, 4, 0, 0.0, True)
    assert_row(b, 5, 5, 0, 0.0, False)
    assert_row(b, 6, 6, 0, 1.0, False)

    b.close()


def test_buffer(filename):
    b = b5.Buffer()
    b.create(filename, state_shape=shape, state_dtype=np.uint8)

    b.append(s1, a1, r1, d1, initial=True)
    b.append(s1, a1, r1, d1)
    b.close()

    b.load(filename, mode='a')
    assert np.allclose(s1, b.state[0])
    assert np.allclose(a1, b.action[0])
    assert r1 == b.reward[0]
    assert d1 == b.done[0]
    assert b.num_episodes == 1
    assert b.steps == 2
    assert b.episodes[0] == 0
    assert b.get_epi_len(0) == 2

    b.append(s2, a2, r2, d2, initial=True)
    b.append(s2, a2, r2, d2)
    b.close()
    b.load(filename, mode='a')

    assert np.allclose(s2, b.state[2])
    assert np.allclose(a2, b.action[2])
    assert r2 == b.reward[2]
    assert d2 == b.done[2]
    assert b.num_episodes == 2
    assert b.steps == 4
    assert b.episodes[0] == 0
    assert b.episodes[1] == 2
    assert b.get_epi_len(1) == 2

def test_n_gram():
    pass