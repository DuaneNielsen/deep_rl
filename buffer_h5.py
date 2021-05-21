import h5py as h5
import io
import numpy as np

class Buffer:
    def __init__(self):
        self.f = None
        self.replay = None
        self.session = {}

    def create(self, filename, state_shape, state_dtype, compression=None, compression_opts=None):
        """

        Args:
            filename:
            state_shape: shape of state tensor
            state_dtype:
            compression: 'gzip', None
            compression_opts: 1 - 9

        Returns:

        """
        state_shape_ = (0, *state_shape)
        self.f = h5.File(filename, 'w')
        self.replay = self.f.create_group('replay')
        self.replay.create_dataset('state', state_shape_, dtype=state_dtype,
                                   chunks=(1, 210, 160, 3), maxshape=(None, 210, 160, 3),
                                   compression='gzip', compression_opts=9, shuffle=True)

    def load(self, filename, mode='r'):
        self.f = h5.File(filename, mode=mode)

    def close(self):
        self.f.close()

    @property
    def state(self):
        return self.f['/replay/state']

    def append(self, s):
        size = len(self.state)
        self.state.resize(size + 1, axis=0)
        self.state[size] = s

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
