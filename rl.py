import io
from collections import deque
import time
from statistics import mean, stdev
import numpy as np
import pickle
import warnings
from logs import logger, list_stats
import tables as tb
import os

global_step = 0
global_best_mean_return = -999999999.0
global_best_return_stats = {}
global_render = False
global_test_number = 1


def step(env, policy, buffer, render=False, timing=False, capture_raw=False, **kwargs):
    """
    Transition generator, advances a single transition each iteration

    Args:
        env: gym environment to step
        policy: policy to use, policy(state) -> action
        global_render: calls env render function if True
        capture_raw: logs the raw input into the buffer instead of the processed input
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

    while True:

        meta = {}

        if timing:
            start_t = time.time()

        if done:
            state, state_info = env.reset(), {}
            if capture_raw:
                state_ref = state_buffer.append(env.render('rgb_array'))
            else:
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

        if capture_raw:
            state_p_ref = state_buffer.append(env.render('rgb_array'))
        else:
            state_p_ref = state_buffer.append(state_p)


        if timing:
            step_t = time.time()

        if global_render:
            env.render()
            meta['frame'] = env.render('rgb_array')

        if timing:
            render_t = time.time()

        policy_time += [policy_t - start_t]
        step_time += [step_t - policy_t]
        render_time += [render_t - step_t]

        if global_step % 1000 == 0:
            mean_policy_time = mean(policy_time) * 1000
            mean_step_time = mean(step_time) * 1000
            mean_render_time = mean(render_time) * 1000
            meta['render_time (ms)'] = mean_render_time
            meta['step_time (ms)'] = mean_step_time
            meta['policy_time (ms)'] = mean_policy_time

            step_time, render_time, policy_time = [], [], []

        yield global_step, state_ref, action, state_p_ref, reward, done, info, meta

        state = state_p
        state_ref = state_p_ref
        done = done
        global_step += 1
        epi_len += 1
        epi_returns += reward

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


"""
On disk replay buffer using Pytables
"""


class OnDiskStateRef:
    def __init__(self, fileh, ref):
        self.fileh = fileh
        self.ref = ref

    def __equal__(self, b):
        return self.ref == b.ref

    def state(self):
        return self.fileh.root.replay.States[self.ref]


class OnDiskStateBuffer:
    def __init__(self, buffer):
        self.buffer = buffer

    @staticmethod
    def create(fileh, expectedrows, shape, dtype, complevel):
        """

        Args:
            state_shape: tuple of state)times
            state_type:

        Returns:

        """
        _shape = (0, *shape)
        atom = tb.Atom.from_dtype(np.dtype(dtype))
        filters = tb.Filters(complevel=complevel, complib='zlib')
        fileh.create_earray('/replay', name='States', atom=atom, shape=_shape,
                                 title="replay: States", filters=filters, expectedrows=expectedrows)

    def append(self, state):
        expected_shape = self.buffer.fileh.root.replay.States.shape[1:]
        assert state.shape == expected_shape, f"expected shape {expected_shape} but state has {state.shape}"
        state = state[np.newaxis, ...]
        self.buffer.fileh.root.replay.States.append(state)
        return StateRef(self, len(self.buffer.fileh.root.replay.States) - 1)

    def __getitem__(self, item):
        return self.buffer.fileh.root.replay.States[item]

    def __len__(self):
        return len(self.buffer.fileh.root.replay.States)


class Transition(tb.IsDescription):
    state = tb.UInt64Col(pos=0)  # pointer to statebuffer
    action = tb.UInt32Col(pos=1)  # action (to be relpaced with pointer in future)
    reward = tb.FloatCol(pos=2)  # reward
    next_state = tb.UInt64Col(pos=3)  # pointer to statebuffer
    done = tb.BoolCol(pos=4)  # true if terminal transition
    episode = tb.UInt64Col(pos=0)  # pointer to episode
    split = tb.UInt64Col(pos=0)  # reserved for splitting the dataset


class Episode(tb.IsDescription):
    initial = tb.UInt64Col()
    terminal = tb.UInt64Col()


class Episodes:
    def __init__(self, buffer):
        self.buffer = buffer

    def __getitem__(self, item):
        return slice(*self.buffer.fileh.root.replay.Episodes[item])

    def __len__(self):
        return len(self.buffer.fileh.root.replay.Episodes)


class OnDiskReplayBuffer:
    def __init__(self):
        self.fileh = None
        self.statebuffer = OnDiskStateBuffer(self)
        self.episodes = Episodes(self)
        self.prev_done = True
        self.split = None

    @property
    def transitions(self):
        return self.fileh.root.replay.Transitions

    @property
    def states(self):
        return self.fileh.root.replay.States

    @staticmethod
    def load(filename):
        assert os.path.isfile(filename), f"{filename} does not exist"
        buffer = OnDiskReplayBuffer()
        buffer.fileh = tb.open_file(filename, mode='a')
        return buffer

    @staticmethod
    def load_random_splits(filename, lengths):
        """
        returns non overlapping random splits of the dataset
        filename: the filename
        :param lengths: list of lengths of each split, [10, 4, 3] 10 is length of split 0, etc...
        :return: None
        """
        assert os.path.isfile(filename), f"{filename} does not exist"
        fileh = tb.open_file(filename, mode='r')
        assert sum(lengths) == len(fileh.root.replay.Transitions), f'total {sum(lengths)} must equal number of transitions ' \
                                                        f'{len(fileh.root.replay.Transitions)}'

        indices = np.random.permutation(sum(lengths))
        buffers = []
        offset = 0
        for l in lengths:
            buffer = OnDiskReplayBuffer()
            buffer.fileh = fileh
            buffer.split = indices[offset:offset + l]
            buffers += [buffer]
            offset += l
        return tuple(buffers)

    @staticmethod
    def load_splits(filename):
        """

        :param filename:
        :return: the splits saved on disk
        """
        raise NotImplementedError

    @staticmethod
    def create(filename, state_shape, state_dtype, expectedrows=1000000, state_complevel=5):
        """

        Args:
            filename: path to the file to store the replaybuffer
            state_shape: dimensions of the state space
            state_dtype: numpy dtype of the state space
            expectedrows: expected number of transitions in the table, used to optimize performance
            state_complevel: zlib compression level, 1 (fastest) -> 9 (smallest)

        Returns:

        """
        assert not os.path.isfile(filename), f"{filename} already exists"
        buffer = OnDiskReplayBuffer()
        fileh = tb.open_file(filename, mode='w')
        buffer.fileh = fileh
        fileh.create_group(fileh.root, "replay")
        fileh.create_table("/replay", "Transitions", Transition, "replay: Transitions", expectedrows=expectedrows)
        fileh.create_table("/replay", "Episodes", Episode, "replay: Episodes", expectedrows=expectedrows)
        OnDiskStateBuffer.create(fileh, expectedrows, state_shape, state_dtype, state_complevel)
        return buffer

    def close(self):
        self.fileh.close()

    def append(self, s, a, s_p, r, d, i):
        epi_row = self.fileh.root.replay.Episodes.row
        row = self.fileh.root.replay.Transitions.row

        # if previous trajectory ended then append and start new one
        if self.prev_done:
            epi_row['initial'] = len(self.fileh.root.replay.Transitions)
        row['state'] = s.ref
        row['action'] = a
        row['next_state'] = s_p.ref
        row['reward'] = r
        row['done'] = d
        row['split'] = 0
        row['episode'] = len(self.fileh.root.replay.Episodes)
        row.append()
        self.fileh.root.replay.Transitions.flush()

        epi_row['terminal'] = len(self.fileh.root.replay.Transitions)
        if d:
            epi_row.append()

        self.fileh.root.replay.Episodes.flush()

        self.prev_done = d

    def make_transition(self, trans):
        """
        Override this method to return what you want for downstream processing

        Args:
            trans: transition

        Returns: data for downstream processing

        """
        s = self.fileh.root.replay.States[trans['state']]
        s_p = self.fileh.root.replay.States[trans['next_state']]
        return s, trans['action'], s_p, trans['reward'], trans['done']

    def gettransition(self, item):
        transitions = self.fileh.root.replay.Transitions[item]
        if isinstance(item, slice):
            return [self.make_transition(trans) for trans in transitions]
        else:
            return self.make_transition(transitions)

    def __len__(self):
        if self.split is None:
            return len(self.fileh.root.replay.Transitions)
        else:
            return len(self.split)

    def __getitem__(self, item):
        if self.split is None:
            return self.gettransition(item)
        else:
            return self.gettransition(self.split[item])