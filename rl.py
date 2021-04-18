import io
from collections import deque
import time
from statistics import mean, stdev
import numpy as np
import pickle

global_step = 0
global_best_mean_return = -999999999.0
global_best_stdev_return = 0


def step(env, policy, buffer, render=False, timing=False, **kwargs):
    """
    Transition generator, advances a single transition each iteration

    Args:
        env: gym environment to step
        policy: policy to use, policy(state) -> action
        render: calls env render function if True
        timing: prints timing info to stdout if True
        kwargs: will be passed to the policy
    """
    global global_step
    state_buffer = buffer.statebuffer
    step_time, render_time, policy_time = [], [], []
    step_t, start_t, policy_t, render_t = 0, 0, 0, 0
    done = True
    state = None
    state_ref = None
    epi_returns = 0
    epi_len = 0
    global_step = 0
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
            if render:
                env.render()

        action = policy(state, **kwargs)

        if timing:
            policy_t = time.time()

        state_p, reward, done, info = env.step(action)
        state_p_ref = state_buffer.append(state_p)

        if timing:
            step_t = time.time()

        if render:
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

    if render:
        env.render()

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


def evaluate(env, policy, sample_n=10, render=False, capture=False):
    """
    Evaluate the policy and save if improved

    Args:
        policy: policy(state) -> action to evaluate
        run_dir: directory to save parameters to
        params: dict of parameters to write..

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

    global global_best_mean_return, global_best_stdev_return

    stats = {}
    start_t = time.time()

    returns = []
    vidstream = []

    for _ in range(sample_n):
        retn, length, video = episode(env, policy, render=render, capture=capture)
        returns.append(retn)
        if capture:
            vidstream.extend(video)

    mean_return = mean(returns)
    stdev_return = stdev(returns)

    end_t = time.time()
    total_t = end_t - start_t

    # checkpoint policy if mean return is better
    if mean_return > global_best_mean_return:
        global_best_mean_return = mean_return
        global_best_stdev_return = stdev_return
        stats["best"] = True
    else:
        stats["best"] = False

    if capture:
        stats["video"] = vidstream

    stats["last_mean_return"] = mean_return
    stats["last_stdev_return"] = stdev_return
    stats["best_mean_return"] = global_best_mean_return
    stats["best_stdev_return"] = global_best_stdev_return
    stats["test_returns"] = returns
    stats["test_mean_return"] = mean_return
    stats["test_wall_time"] = total_t

    return stats


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
        return StateRef(self, len(self.states)-1)

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
        return StateRef(self, len(self.states)-1)

    def __getitem__(self, item):
        return self.decode(self.states[item])

    def __len__(self):
        return len(self.states)


def encode_z(A):
    compressed_array = io.BytesIO()    # np.savez_compressed() requires a file-like object to write to
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

    def append(self, s, a, s_p, r, d):
        self.buffer.append((s, a, s_p, r, d))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        s, a, s_p, r, d = self.buffer[item]
        return s.state(), a, s_p.state(), r, d