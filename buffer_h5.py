import h5py as h5
import numpy as np
import time
from statistics import mean
import rich.table
from rich import print


class Driver:
    def __init__(self):
        self.step = 0


class Column:
    def __init__(self, name, shape=None, dtype=None, chunk_size=1, compression=None, compression_opts=None, shuffle=True):
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
                             compression=self.compression, compression_opts=self.compression_opts, shuffle=self.shuffle)

    # def append(self, group, name, step, data):
    #     if step % chunk_size[0] == 0:
    #         resized = step + self.chunk_size
    #         group[name].resize(resized, axis=0)
    #     group[name][step] = data


class Buffer:
    def __init__(self):
        self.f = None
        self.replay = None
        self.chunk_size = 1000000
        self.n_gram_index = None
        self.run_step = 0

    def create(self, filename, columns):
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

        self.replay = self.f.create_group('replay')

        self.replay.attrs.create('steps', 0)
        self.replay.attrs.create('episodes', 0)

        reward_col = Column('reward', dtype=np.float32, chunk_size=self.chunk_size)
        done_col = Column('done', dtype=np.bool_, chunk_size=self.chunk_size)
        episodes_col = Column('episodes', dtype=np.int64, chunk_size=self.chunk_size)
        columns += [reward_col, done_col, episodes_col]

        for column in columns:
            column.create(self.replay)

    def load(self, filename, mode='r'):
        self.f = h5.File(filename, mode=mode)

    def close(self):
        self.f.close()

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

    def get_epi_len(self, item, gram_len=2):
        """
        Number of grams in a episode
        Args:
            item: episode index
            gram_len: length of gram

        Returns: the number of grams in an episode,
        to get number of steps set gram_len = 1

        """
        if item < self.num_episodes - 1:
            return self.episodes[item + 1] - self.episodes[item] - gram_len + 1
        else:
            return self.steps - self.episodes[self.num_episodes - 1] - gram_len + 1

    def append(self, state, action, reward, done, initial=False, **kwargs):

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

    def n_gram(self, item, gram_len=2):
        if self.n_gram_index is None:
            self.n_gram_index = self.make_n_gram_index(gram_len)
        i = self.n_gram_index[item]
        gram = slice(i + 1 - gram_len, i + 1)
        s = self.state[gram]
        a = self.action[gram]
        r = self.reward[gram]
        d = self.done[gram]
        return s, a, r, d

    def print_stats(self):
        table = rich.table.Table(title=f"{self.__class__.__name__}")
        table.add_column("Stat", justify="right", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_row("File", f'{self.f.filename}')
        table.add_row("Episodes", f"{self.num_episodes}")
        epi_lengths = [self.get_epi_len(self.episodes[e]) for e in range(self.num_episodes)]
        table.add_row("Mean episode len", f"{mean(epi_lengths)}")
        table.add_row("Transitions", f"{self.n_gram_len(gram_len=2)}")
        # table.add_row("Transitions with + reward", f"{len(self.transitions.get_where_list('reward > 0'))}")
        # table.add_row("Transitions with - reward", f"{len(self.transitions.get_where_list('reward < 0'))}")
        # table.add_row("Transitions with 0 reward", f"{len(self.transitions.get_where_list('reward == 0'))}")
        print(table)

    def __len__(self):
        return self.n_gram_len(gram_len=2)

    def __getitem__(self, item):
        g = self.n_gram(item, gram_len=2)
        s, a, s_p, r, d = g[0][0], g[1][1], g[1][0], g[1][2], g[1][3]
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
        step_time, render_time, policy_time = [], [], []
        step_t, start_t, policy_t, render_t = 0, 0, 0, 0
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

            self.append(state_p, action, reward, done, raw=raw_state)

            if timing:
                step_t = time.time()

            if render:
                env.render()
                meta['frame'] = env.render('rgb_array')

            if timing:
                render_t = time.time()

            policy_time += [policy_t - start_t]
            step_time += [step_t - policy_t]
            render_time += [render_t - step_t]

            if self.run_step % 1000 == 0:
                mean_policy_time = mean(policy_time) * 1000
                mean_step_time = mean(step_time) * 1000
                mean_render_time = mean(render_time) * 1000
                meta['render_time (ms)'] = mean_render_time
                meta['step_time (ms)'] = mean_step_time
                meta['policy_time (ms)'] = mean_policy_time

                step_time, render_time, policy_time = [], [], []

            yield self.run_step, state, action, state_p, reward, done, info, meta

            state = state_p
            done = done
            self.run_step += 1
            epi_len += 1
            epi_returns += reward
