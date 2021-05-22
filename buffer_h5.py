import h5py as h5
import numpy as np


class Buffer:
    def __init__(self):
        self.f = None
        self.replay = None
        self.chunk_size = 1000000
        self.n_gram_index = None

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

        self.replay.create_dataset('episodes', scalar_shape_, dtype=np.bool_,
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

    def append(self, s, a, r, d, initial=False):

        if self.steps % self.chunk_size == 0:
            resized = self.steps + self.chunk_size
            self.action.resize(resized, axis=0)
            self.reward.resize(resized, axis=0)
            self.done.resize(resized, axis=0)
        self.state.resize(self.steps + 1, axis=0)

        self.state[self.steps] = s
        self.action[self.steps] = a
        self.reward[self.steps] = r
        self.done[self.steps] = d

        if initial:
            if self.num_episodes % self.chunk_size == 0:
                self.episodes.resize(self.num_episodes + self.chunk_size, axis=0)

            self.episodes[self.num_episodes] = self.steps
            self.num_episodes += 1

        self.steps += 1

    def n_gram_len(self, size=2):
        return self.steps - self.num_episodes * size

    def get_epi_len(self, item):
        if item < self.num_episodes-1:
            return self.episodes[item+1] - self.episodes[item]
        else:
            return self.steps - self.episodes[self.num_episodes-1]

    def get_n_gram_index(self, size=2):
        index = []
        for e in range(self.num_episodes):
            start = self.episodes[e]
            end = self.get_epi_len(e) + start
            index += [range(start + size - 1, end)]
        return index

    def n_gram(self, item, size=2):
        i = self.n_gram_index[item]
        gram = slice(i+1-size, i+1)
        s = self.state[gram]
        a = self.action[gram]
        r = self.reward[gram]
        d = self.done[gram]
        return s, a, r, d


# bio = io.BytesIO()
# with h5.File(bio, 'w') as f:
#     f['dataset'] = range(10)
#
#     data = bio.getvalue()
#     print(f'Total_size {len(data)}')
#     print("First bytes:", data[:10])
#     grp = f.create_group('replay')
#     print(f.name)
#     print(grp.name)
#
#     states =
#     image = np.random.randint((210, 160, 3), dtype=np.uint8)
#     states[0] = image
#     f.close()
#
# with h5.File(bio, 'r') as f:
#     print(f['/replay/state'][0])
