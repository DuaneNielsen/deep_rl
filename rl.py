import io
from collections import deque
import time
from statistics import mean, stdev
import numpy as np
import pickle
import warnings
from logs import logger, list_stats

global_step = 0
global_best_mean_return = -999999999.0
global_best_return_stats = {}
global_render = False
global_test_number = 1


def step(env, policy, buffer, render=False, timing=False, **kwargs):
    """
    Transition generator, advances a single transition each iteration

    Args:
        env: gym environment to step
        policy: policy to use, policy(state) -> action
        global_render: calls env render function if True
        timing: prints timing info to stdout if True
        kwargs: will be passed to the policy
    """
    global global_step, global_render
    state_buffer = buffer.statebuffer
    step_time, render_time, policy_time = [], [], []
    step_t, start_t, policy_t, render_t = 0, 0, 0, 0
    done = True
    state = None
    state_ref = None
    epi_returns = 0
    epi_len = 0
    global_step = 0
    global_render = render
    meta = {}

    while True:

        if timing:
            start_t = time.time()

        if done:
            state, state_info = env.reset(), {}
            state_ref = state_buffer.append(state)
            meta['epi_returns'] = epi_returns
            meta['epi_len'] = epi_len
            epi_returns = 0
            epi_len = 0
            if global_render:
                env.render()

        action = policy(state, **kwargs)

        if timing:
            policy_t = time.time()

        state_p, reward, done, info = env.step(action)
        state_p_ref = state_buffer.append(state_p)

        if timing:
            step_t = time.time()

        if global_render:
            env.render()
            meta['frame'] = env.render('rgb_array')

        if timing:
            render_t = time.time()

        yield global_step, state_ref, action, state_p_ref, reward, done, info, meta

        state = state_p
        state_ref = state_p_ref
        done = done
        global_step += 1
        epi_len += 1
        epi_returns += reward

        policy_time += [policy_t - start_t]
        step_time += [step_t - policy_t]
        render_time += [render_t - step_t]

        if global_step % 1000 == 0:
            mean_policy_time = mean(policy_time) * 1000
            mean_step_time = mean(step_time) * 1000
            mean_render_time = mean(render_time) * 1000

            if mean_render_time > 1.0:
                meta['render_time (ms)'] = mean_render_time
            if mean_step_time > 1.0:
                meta['step_time (ms)'] = mean_step_time
            if mean_policy_time > 1.0:
                meta['policy_time (ms)'] = mean_policy_time

            step_time, render_time, policy_time = [], [], []


def _render_env(env, render, delay):
    if render:
        env.render()
        time.sleep(delay)


def episode(env, policy, render=False, delay=0.01, capture=False):
    """
    Runs one episode using the provided policy on the environment

    Args:
        env: gym environment to generate an episode for
        policy: policy(state) -> action takes state as input, must output an action runnable on the environment
        render: if True will call environments render function
        delay: rendering delay
        kwargs: kwargs will be passed to policy
    """

    returns = 0
    length = 0
    vid_buffer = []

    try:
        if render:
            env.render()
    except AttributeError:
        pass  # some environments do not like rendering before reset, but some require it

    state, reward, done, info = env.reset(), 0.0, False, {}
    action = policy(state)
    _render_env(env, render, delay)

    if capture:
        frame = env.render(mode='rgb_array')
        vid_buffer.append(frame)

    while not done:
        state, reward, done, info = env.step(action)
        returns += reward
        length += 1
        action = policy(state)
        _render_env(env, render, delay)
        if capture:
            frame = env.render(mode='rgb_array')
            vid_buffer.append(frame)

    return returns, length, vid_buffer


def demo(demo, env, policy):
    """
    Infinite loop that runs policy on environment, and renders
    Args:
        demo: if true enters infinite loop
        policy: policy(state) -> action to run

    """
    if demo:
        while True:
            episode(env, policy, render=True)


def evaluate(env, policy, sample_n=10, vid_sample_n=0):
    """
    Evaluate the policy and save if improved

    Args:
        env: gym env to avaluate on
        policy: policy(state) -> action to evaluate
        sample_n: number of episodes to sample
        capture: capture video
        vid_sample_n: number of episodes to capture video for

        .. code-block:: python
          policy_net = nn.Linear(4, 1)
          optim = nn.optim.SGD(policy_net.parameters(), lr=1e-4)
          params={'policy_net': policy_net, 'optim': optim}

        sample_n: number of episodes to sample
        render: render the environment while running if True

    Returns:
        mean_return
        stdev_return

    """

    global global_best_mean_return, global_best_return_stats, global_test_number

    start_t = time.time()

    lengths = []
    returns = []
    vidstream = []

    for n in range(sample_n):
        capture = n < vid_sample_n
        retn, length, video = episode(env, policy, capture=capture)
        returns.append(retn)
        lengths.append(length)
        vidstream.extend(video)

    return_stats = list_stats("eval-return", returns)

    end_t = time.time()
    total_t = end_t - start_t

    """ if policy improved save it's stats"""
    if return_stats['eval-return Mean'] > global_best_mean_return:
        global_best_mean_return = return_stats['eval-return Mean']
        for key, value in return_stats.items():
            key = key.replace('eval-return', 'eval-best-return')
            global_best_return_stats[key] = value
        improved = True
    else:
        improved = False

    if len(vidstream) > 0:
        vidstream = np.stack(vidstream)
        if not (vidstream.shape[3] == 3 or vidstream.shape[3] == 1):
            warnings.warn(f'Vidstream shape is {vidstream.shape} but T, H, W, C format was expected')
        logger.log[f"video_{global_test_number}"] = vidstream

    logger.log.update(return_stats)
    logger.log.update(global_best_return_stats)
    logger.log.update(list_stats("eval-lengths", lengths))
    logger.log["eval-wall_time"] = total_t

    return improved


def save(buffer, filepath):
    file = open(filepath, mode='wb')
    pickle.dump(buffer, file)
    file.close()


def load(filepath):
    file = open(filepath, 'rb')
    load_buff = pickle.load(file)
    file.close()
    return load_buff


class StateRef:
    def __init__(self, buffer, ref):
        self.buffer = buffer
        self.ref = ref

    def state(self):
        return self.buffer[self.ref]


class StateBuffer:
    def __init__(self):
        self.states = []

    def append(self, item):
        self.states.append(item)
        return StateRef(self, len(self.states) - 1)

    def __getitem__(self, item):
        return self.states[item]

    def __len__(self):
        return len(self.states)


class EncodedStateBuffer:
    def __init__(self, encode, decode):
        self.states = []
        self.encode = encode
        self.decode = decode

    def append(self, item):
        self.states.append(self.encode(item))
        return StateRef(self, len(self.states) - 1)

    def __getitem__(self, item):
        return self.decode(self.states[item])

    def __len__(self):
        return len(self.states)


def encode_z(A):
    compressed_array = io.BytesIO()  # np.savez_compressed() requires a file-like object to write to
    np.savez_compressed(compressed_array, A)
    return compressed_array


def decode_z(compressed_array):
    compressed_array.seek(0)  # seek back to the beginning of the file-like object
    return np.load(compressed_array)['arr_0']


class ZCompressedStateBuffer(EncodedStateBuffer):
    def __init__(self):
        super().__init__(encode_z, decode_z)


class ReplayBuffer:
    def __init__(self, maxlen=None, compression=None):
        self.buffer = deque(maxlen=maxlen)
        self.statebuffer = compression if compression is not None else StateBuffer()
        self.episodes = []
        self.epi_start = 0

    def append(self, s, a, s_p, r, d):
        self.buffer.append((s, a, s_p, r, d))
        if d:
            self.episodes.append((self.epi_start, len(self.buffer)))
            self.epi_start = len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        s, a, s_p, r, d = self.buffer[item]
        return s.state(), a, s_p.state(), r, d


class TopTrajectoryDataset:
    def __init__(self, replay_buffer, min_return):
        self.replay_buffer = replay_buffer
        self.index = []

        for start, end in self.replay_buffer.episodes:
            epi_return = 0
            for i in range(start, end):
                epi_return += self.replay_buffer[i][3]
            if epi_return >= min_return:
                self.index += list(range(start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        return self.replay_buffer[self.index[item]]
