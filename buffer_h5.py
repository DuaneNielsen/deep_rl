import h5py as h5
import numpy as np
import time
from statistics import mean


class Driver:
    def __init__(self):
        self.step = 0


class Buffer:
    def __init__(self):
        self.f = None
        self.replay = None
        self.chunk_size = 1000000
        self.n_gram_index = None
        self.run_step = 0

    def create(self, filename,
               state_shape, state_dtype, compression=None, compression_opts=None,
               action_shape=None, action_dtype=None):
        """

        Args:
            filename:
            state_shape: shape of state tensor
            state_dtype:
            compression: 'gzip', None
            compression_opts: 1 - 9

        Returns:

        """

        def make_scm_tuples(shape, chunk_size):
            return (0, *shape), (chunk_size, *shape), (None, *shape)

        state_shape_, state_chunks_, state_max_ = make_scm_tuples(state_shape, 1)

        if action_shape is None:
            action_shape_, action_chunks_, action_max_ = (0, ), (self.chunk_size, ), (None, )
        else:
            action_shape_, action_chunks_, action_max_ = make_scm_tuples(action_shape, self.chunk_size)

        scalar_shape_, scalar_chunks_, scalar_max_ = (0, ), (self.chunk_size, ), (None, )

        self.f = h5.File(filename, 'w')

        self.replay = self.f.create_group('replay')

        self.replay.attrs.create('steps', 0)
        self.replay.attrs.create('episodes', 0)

        self.replay.create_dataset('state', state_shape_, dtype=state_dtype,
                                   chunks=state_chunks_, maxshape=state_max_,
                                   compression=compression, compression_opts=compression_opts, shuffle=True)
        self.replay.create_dataset('action', action_shape_, dtype=action_dtype,
                                   chunks=action_chunks_, maxshape=action_max_,
                                   compression='gzip', compression_opts=3, shuffle=True)
        self.replay.create_dataset('reward', scalar_shape_, dtype=np.float32,
                                   chunks=scalar_chunks_, maxshape=scalar_max_,
                                   compression='gzip', compression_opts=3, shuffle=True)
        self.replay.create_dataset('done', scalar_shape_, dtype=np.bool_,
                                   chunks=scalar_chunks_, maxshape=scalar_max_,
                                   compression='gzip', compression_opts=3, shuffle=True)

        self.replay.create_dataset('episodes', scalar_shape_, dtype=np.int64,
                                   chunks=scalar_chunks_, maxshape=scalar_max_,
                                   compression='gzip', compression_opts=3, shuffle=True)

    def load(self, filename, mode='r'):
        self.f = h5.File(filename, mode=mode)

    def close(self):
        self.f.close()

    @property
    def state(self):
        return self.f['/replay/state']

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

    def get_epi_len(self, item):
        if item < self.num_episodes-1:
            return self.episodes[item+1] - self.episodes[item]
        else:
            return self.steps - self.episodes[self.num_episodes-1]

    def append(self, s, a, r, d, initial=False):

        if self.steps % self.chunk_size == 0:
            resized = self.steps + self.chunk_size
            self.action.resize(resized, axis=0)
            self.reward.resize(resized, axis=0)
            self.done.resize(resized, axis=0)
        self.state.resize(self.steps + 1, axis=0)

        self.state[self.steps] = s

        if a is None:
            if len(self.action.shape) > 1:
                a = np.zeros(self.action.shape[1:], dtype=self.action.dtype)
            else:
                a = np.zeros(1, dtype=self.action.dtype)
        self.action[self.steps] = a

        self.reward[self.steps] = r
        self.done[self.steps] = d

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
            end = self.get_epi_len(e) + start
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

    def __len__(self):
        return self.n_gram_len(gram_len=2)

    def __getitem__(self, item):
        g = self.n_gram(item, gram_len=2)
        s, a, s_p, r, d = g[0][0], g[1][1], g[1][0], g[1][2], g[1][3]
        return s, a, s_p, r, d

    def step(self, env, policy, buffer, render=False, timing=False, capture_raw=False, **kwargs):
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
                    state = env.render('rgb_array')
                meta['epi_returns'] = epi_returns
                meta['epi_len'] = epi_len
                epi_returns = 0
                epi_len = 0
                if render:
                    env.render()

                buffer.append(state, None, 0.0, False, initial=True)

            action = policy(state, **kwargs)

            if timing:
                policy_t = time.time()

            state_p, reward, done, info = env.step(action)

            if capture_raw:
                state = env.render('rgb_array')

            buffer.append(state_p, action, reward, done)

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