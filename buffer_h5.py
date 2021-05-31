import h5py as h5
import numpy as np
import time
from statistics import mean
import rich.table
from rich.progress import track
from rich import print
import argparse
from abc import ABC


class Column(ABC):

    def create(self, group):
        raise NotImplementedError


class NumpyColumn(Column):
    def __init__(self, name, shape=None, dtype=None, chunk_size=1, compression=None, compression_opts=None,
                 shuffle=True):
        self.shape = shape
        self.name = name
        self.dtype = dtype
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.chunk_size = chunk_size

    def make_scm_tuples(self):
        if self.shape is not None:
            return (0, *self.shape), (self.chunk_size, *self.shape), (None, *self.shape)
        else:
            return (0,), (self.chunk_size,), (None,)

    def create(self, group):
        shape, chunkshape, maxshape = self.make_scm_tuples()
        group.create_dataset(self.name, shape, dtype=self.dtype,
                             chunks=chunkshape, maxshape=maxshape,
                             compression=self.compression, compression_opts=self.compression_opts, shuffle=False)


class Buffer:
    def __init__(self):
        """
        Append is expected to be sequential
        Simultaneous reading and writing not really supported (poor perforrmance or might not work)
        """
        self.f = None
        self.chunk_size = 1000000
        self.n_gram_index = None
        self.run_step = 0
        self._episode_len = None
        self.dirty = True

    def create(self, filename, state_col, action_col=None, raw_col=None):
        # todo add metadata for size and type of action space
        """

        Args:
            filename:
            state_shape: shape of state tensor
            state_dtype:
            compression: 'gzip', None
            compression_opts: 1 - 9

        Returns:

        """

        self.f = h5.File(filename, 'w')

        self.f.create_group('replay')

        self.replay.attrs.create('steps', 0)
        self.replay.attrs.create('episodes', 0)

        # force the names to be sure
        state_col.name = 'state'
        if action_col is None:
            action_col = NumpyColumn('action', dtype=np.int64)
        else:
            action_col.name = 'action'

        columns = [state_col, action_col]

        # raw col is optional
        if raw_col is not None:
            raw_col.name = 'raw'
            columns.append(raw_col)

        # the below columns are not optional
        reward_col = NumpyColumn('reward', dtype=np.float32, chunk_size=self.chunk_size)
        done_col = NumpyColumn('done', dtype=np.bool_, chunk_size=self.chunk_size)
        episodes_col = NumpyColumn('episodes', dtype=np.int64, chunk_size=self.chunk_size)
        columns += [reward_col, done_col, episodes_col]

        for column in columns:
            column.create(self.replay)

    def load(self, filename, mode='r', cache_bytes=1073741824, cache_slots=100000, cache_w0=0.0, **kwargs):
        self.f = h5.File(filename, mode=mode, rdcc_nbytes=cache_bytes, rdcc_nslots=cache_slots, rdcc_w0=cache_w0)

    def close(self):
        self.f.close()

    @property
    def replay(self):
        return self.f['/replay']

    @property
    def state(self):
        return self.f['/replay/state']

    @property
    def raw(self):
        return self.f['/replay/raw']

    @property
    def action(self):
        return self.f['/replay/action']

    @property
    def reward(self):
        return self.f['/replay/reward']

    @property
    def done(self):
        return self.f['/replay/done']

    @property
    def attrs(self):
        return self.f['/replay'].attrs

    @property
    def steps(self):
        return self.attrs['steps']

    @steps.setter
    def steps(self, item):
        self.attrs['steps'] = item

    @property
    def num_episodes(self):
        return self.attrs['episodes']

    @num_episodes.setter
    def num_episodes(self, item):
        self.attrs['episodes'] = item

    @property
    def episodes(self):
        return self.f['/replay/episodes']

    def clean(self):
        self.compute_1_gram_epi_lengths()
        self.dirty = False

    def compute_1_gram_epi_lengths(self):
        epi_start = self.episodes[:self.num_episodes]
        epi_end = np.roll(epi_start.copy(), shift=-1, axis=0)
        self._episode_len = epi_end - epi_start
        self._episode_len[-1] = self.steps - epi_start[-1]

    @property
    def episode_len(self):
        if self.dirty:
            self.clean()
        return self._episode_len

    def get_epi_len(self, item_or_slice, gram_len=2):
        """
        Number of grams in a episode
        Args:
            item: episode index
            gram_len: length of gram

        Returns: the number of grams in an episode,
        to get number of steps set gram_len = 1

        """
        return self.episode_len[item_or_slice] - gram_len + 1

    def append(self, state, action, reward, done, initial=False, **kwargs):
        self.dirty = True

        if self.steps % self.chunk_size == 0:
            resized = self.steps + self.chunk_size
            self.action.resize(resized, axis=0)
            self.reward.resize(resized, axis=0)
            self.done.resize(resized, axis=0)

        self.state.resize(self.steps + 1, axis=0)
        self.state[self.steps] = state

        if 'raw' in kwargs:
            if kwargs['raw'] is not None:
                self.replay['raw'].resize(self.steps + 1, axis=0)
                self.replay['raw'][self.steps] = kwargs['raw']

        if action is None:
            if len(self.action.shape) > 1:
                action = np.zeros(self.action.shape[1:], dtype=self.action.dtype)
            else:
                action = np.zeros(1, dtype=self.action.dtype)
        self.action[self.steps] = action

        self.reward[self.steps] = reward
        self.done[self.steps] = done

        if initial:
            if self.num_episodes % self.chunk_size == 0:
                self.episodes.resize(self.num_episodes + self.chunk_size, axis=0)

            self.episodes[self.num_episodes] = self.steps
            self.num_episodes += 1

        self.steps += 1

    def n_gram_len(self, gram_len=2):
        return self.steps - self.num_episodes * (gram_len - 1)

    def make_n_gram_index(self, gram_len):
        index = []
        for e in range(self.num_episodes):
            start = self.episodes[e]
            end = self.get_epi_len(e, gram_len=1) + start
            if start < end:
                index += range(start + gram_len - 1, end)
        return index

    def n_gram(self, item, gram_len=2, fields=None):
        """

        Args:
            item: the index of the gram
            gram_len: the n in n-gram, a transition is a 2-gram, defaults to 2-gram
            fields: list of fields to return, default ['state', 'action', 'reward', 'done']

        Returns: tuple in order of the fields, gram is the first index, if 1 gram first index is omitted,
        order of gram is same order as append sequence

        """
        fields = ['state', 'action', 'reward', 'done'] if fields is None else fields
        if self.n_gram_index is None:
            self.n_gram_index = self.make_n_gram_index(gram_len)
        i = self.n_gram_index[item]
        gram = slice(i + 1 - gram_len, i + 1)
        results = []
        for field in fields:
            results.append(self.replay[field][gram])
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    def make_stat_table(self):
        table = rich.table.Table(title=f"{self.__class__.__name__}")
        table.add_column("Stat", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("File", f'{self.f.filename}')
        table.add_row("Episodes", f"{self.num_episodes}")
        epi_lengths = self.episode_len - 1  # subtrct one because we measure in 2 grams
        table.add_row("Mean episode len", f"{np.mean(epi_lengths):.0f}")
        table.add_row("Max episode len", f"{epi_lengths.max()}")
        table.add_row("Min episode len", f"{epi_lengths.min()}")
        table.add_row("Transitions", f"{self.n_gram_len(gram_len=2)}")
        table.add_row("Transitions with + reward", f"{np.count_nonzero(self.reward[:self.steps] > 0.0)}")
        table.add_row("Transitions with - reward", f"{np.count_nonzero(self.reward[:self.steps] < 0.0)}")
        table.add_row("Transitions with 0 reward", f"{np.count_nonzero(self.reward[:self.steps] == 0.0)}")
        return table

    def print_stats(self):
        table = self.make_stat_table()
        print(table)

    def __len__(self):
        return self.n_gram_len(gram_len=2)

    def __getitem__(self, item):
        g = self.n_gram(item, gram_len=2)
        s, a, s_p, r, d = g[0][0], g[1][1], g[0][1], g[2][1], g[3][1]
        return s, a, s_p, r, d

    def step(self, env, policy, render=False, timing=False, capture_raw=False, **kwargs):
        """
        Transition generator, advances a single transition each iteration

        Args:
            env: gym environment to step
            policy: policy to use, policy(state) -> action
            render: calls env render function if True
            capture_raw: logs the raw input into the buffer instead of the processed input
            timing: prints timing info to stdout if True
            kwargs: will be passed to the policy
        """
        step_time, render_time, policy_time, append_time = [], [], [], []
        step_t, start_t, policy_t, append_t, render_t = 0, 0, 0, 0, 0
        done = True
        state = None
        epi_returns = 0
        epi_len = 0

        while True:

            meta = {}

            if timing:
                start_t = time.time()

            if done:
                state = env.reset()

                if capture_raw:
                    raw_state = env.render('rgb_array')
                else:
                    raw_state = None

                meta['epi_returns'] = epi_returns
                meta['epi_len'] = epi_len
                epi_returns = 0
                epi_len = 0
                if render:
                    env.render()

                self.append(state, None, 0.0, False, initial=True, raw=raw_state)

            action = policy(state, **kwargs)

            if timing:
                policy_t = time.time()

            state_p, reward, done, info = env.step(action)

            if capture_raw:
                raw_state = env.render('rgb_array')
            else:
                raw_state = None

            if timing:
                step_t = time.time()

            self.append(state_p, action, reward, done, raw=raw_state)

            if timing:
                append_t = time.time()

            if render:
                env.render()
                meta['frame'] = env.render('rgb_array')

            if timing:
                render_t = time.time()
                policy_time += [policy_t - start_t]
                step_time += [step_t - policy_t]
                append_time += [append_t - step_t]
                render_time += [render_t - append_t]

            if self.run_step % 1000 == 0:
                if timing:
                    mean_policy_time = mean(policy_time) * 1000
                    mean_step_time = mean(step_time) * 1000
                    mean_append_time = mean(append_time) * 1000
                    mean_render_time = mean(render_time) * 1000
                    meta['render_time (ms)'] = mean_render_time
                    meta['step_time (ms)'] = mean_step_time
                    meta['append_time (ms)'] = mean_append_time
                    meta['policy_time (ms)'] = mean_policy_time

                    step_time, render_time, policy_time, append_time = [], [], [], []

            yield self.run_step, state, action, state_p, reward, done, info, meta

            state = state_p
            done = done
            self.run_step += 1
            epi_len += 1
            epi_returns += reward


def postprocess_raw(filename, new_column, f, batch_size):
    # not tested
    b = Buffer()
    b.load(filename)
    if new_column.name in b.replay:
        exit()
    new_column.create(b.replay)
    f.replay[new_column.name].resize(b.steps, axis=0)
    num_batches = b.steps // batch_size
    remainder = b.steps % batch_size
    for batch in range(num_batches):
        f.replay[new_column.name][batch:batch + batch_size] = f(b.raw[batch:batch + batch_size])
    if remainder > 0:
        f.replay[new_column.name][-remainder:b.steps] = f(b.raw[-remainder:b.steps])
    b.close()


def grayConversion(image):
    grayValue = 0.07 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.21 * image[:, :, 0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


import cv2


def postprocess_gradient(filename, new_column, debug=False):
    b = Buffer()
    b.load(filename, mode='a')

    # recreate the column if it already exists
    try:
        if '/replay/grad' in b.f:
            del b.f['/replay/grad']
        new_column.create(b.replay)
    except KeyError:
        print('table existed but could not delete')

    b.replay['grad'].resize(b.steps, axis=0)
    for i in track(range(b.steps), description='[blue] writing zeros'):
        b.replay['grad'][i] = np.zeros(new_column.shape, dtype=new_column.dtype)

    for i in track(range(len(b)), description='[blue] adding gradient'):
        raw = b.n_gram(i, fields=['raw'])
        grad = raw[1] - raw[0]
        iw = b.n_gram_index[i]
        b.replay['grad'][iw] = grad
        if debug:
            cv2.imshow('grad', grad)
            cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('postprocessing', choices=['grad'])
    parser.add_argument('filename')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.postprocessing == 'grad':
        grad_col = NumpyColumn('grad', shape=(210, 160, 3), dtype=np.uint8, compression='gzip', compression_opts=9)
        postprocess_gradient(args.filename, grad_col, args.debug)
