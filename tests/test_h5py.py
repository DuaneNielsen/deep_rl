import pytest
import buffer_h5 as b5
import os
import numpy as np


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


def test_buffer(filename):
    b = b5.Buffer()
    b.create(filename, state_shape=shape, state_dtype=np.uint8)
    b.append(s1, a1, r1, d1, initial=True)
    b.append(s1, a1, r1, d1)
    b.close()

    b.load(filename)
    print(list(b.f['/replay'].attrs.keys()))
    assert np.allclose(s1, b.state[0])
    assert np.allclose(a1, b.action[0])
    assert r1 == b.reward[0]
    assert d1 == b.done[0]
    assert b.num_episodes == 1
    assert b.steps == 2
    assert b.episodes[0] == 0
    assert b.get_epi_len(0) == 2