import env
import gymnasium as gym


def test_rock_paper_scissors():
    """

    0: Rock
    1: Paper
    2: Scissors

    """
    test_env = gym.make("IteratedRockPaperScissors-v1")
    state, info = test_env.reset()
    assert state.size == 6

    state, reward, done, terminated, info = test_env.step([0, 1])
    assert reward == -1

    assert state.size == 6
    assert state[0] == 1
    assert state[1] == 2
    assert state[2] == 0
    assert state[3] == 0
    assert state[4] == 0
    assert state[5] == 0

    state, reward, done, terminated, info = test_env.step([1, 2])

    assert reward == -1

    assert state.size == 6
    assert state[0] == 1
    assert state[1] == 2
    assert state[2] == 2
    assert state[3] == 3
    assert state[4] == 0
    assert state[5] == 0

    state, reward, done, terminated, info = test_env.step([0, 1])

    assert reward == -1
    assert done

    assert state.size == 6
    assert state[0] == 4
    assert state[1] == 4
    assert state[2] == 4
    assert state[3] == 4
    assert state[4] == 4
    assert state[5] == 4


def test_rock_paper_scissors_state():
    test_env_0 = gym.make("IteratedRockPaperScissors-v1")
    test_env_1 = gym.make("IteratedRockPaperScissors-v1")
    assert (test_env_1.states() == test_env_0.states()).all()

