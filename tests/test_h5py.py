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
    yield 'test.h5'
    os.remove('test.h5')

shape = (210, 160, 3)
dtype = np.uint8
s1 = np.random.randint(shape, dtype=dtype)


def test_buffer(filename):
    b = b5.Buffer()
    b.create(filename, state_shape=shape, state_dtype=np.uint8)
    b.append(s1)
    b.close()

    b.load(filename)
    assert np.allclose(s1, b.state[0])
