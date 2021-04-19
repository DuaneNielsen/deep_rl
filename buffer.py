from collections import namedtuple
import pickle
import time
from statistics import mean

"""
buffer.py
===================================
Replay buffer based on gym wrappers.  


Just wrap your environment to get a replay buffer.

.. code-block:: python
    
    import gym
    import buffer as bf
    from torch.utils.data import DataLoader
    
    env = gym.make('CartPole-v1')
    env, buffer = bf.wrap(env)
    buffer.enrich(bf.DiscountedReturns(key='g', discount=0.9))
    
    # generate a few transitions
    env.reset()
    env.step(action)
    
    dataset = bf.ReplayBufferDataset(buffer, info_keys=['g'])
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    
    for state, action, state_p, reward, done, returns in dl:
        learn()

"""


FullTransition = namedtuple('FullTransition', ['s', 'a', 's_p', 'r', 'd', 'i'])


class ReplayBuffer:
    def __init__(self):
        """
        Replay buffer

        Attributes:
        buffer          [a, s, r, d, i), () ...]
                        2 transition trajectory would be..
                        [(None, s, 0.0, False, {}), (a, s, r, d, i), (a, s, r, d, i)]
        trajectories    [(start, end), ...]
        transitions     a flat index into buffer that points to the head of each transition
                        ie: to retrieve the n'th transition state, action, state', reward, done, info
                        transition_index = transitions[n]
                        _, state, _, _, _ = buffer[transition_index]
                        action, state_prime, reward, done, info = buffer[transtion_index + 1]
        trajectory_info a dictionary for each trajectory that contains at a minimum
                            return:  The return (total reward) for the trajectory
                            len:  The length of the trajectory
        """
        self.buffer = []
        self.trajectories = []
        self.trajectory_info = []
        self.transitions = []
        self.traj_start = 0
        self._enrich = []
        self.new_trajectory_starting = True
        self.eps_reward = 0
        self.eps_len = 0
        self.record = True

    def clear(self):
        """ clears the buffer """
        self.buffer = []
        self.trajectories = []
        self.transitions = []
        self.trajectory_info = []
        self.traj_start = 0

    def tail_trajectory_complete(self):
        """
        Returns:
            True if the most recent trajectory in the buffer is complete, or if buffer is empty
            False if a trajectory is in the process of being added
        """
        return self.traj_start == len(self.buffer)

    def append(self, s, a, s_p, r, d, i):
        if self.new_trajectory_starting:  # first transition of trajectory
            self.new_trajectory_starting = False
            self.eps_reward = 0
            self.eps_len = 0
            self.buffer.append((None, s, 0.0, False, {}))
            self.transitions.append(len(self.buffer) - 1)

        self.buffer.append((a, s_p, r, d, i))
        self.eps_reward += r
        self.eps_len += 1

        if d:
            """ terminal state, trajectory is complete """
            self.trajectories.append((self.traj_start, len(self.buffer)))
            self.traj_start = len(self.buffer)
            self.trajectory_info.append({'return': self.eps_reward, 'len': self.eps_len})
            self.new_trajectory_starting = True
        else:
            """ if not terminal, then by definition, this will be a transition """
            self.transitions.append(len(self.buffer) - 1)

    def __getitem__(self, item):
        item = self.transitions[item]
        _, s, _, _, i = self.buffer[item]
        a, s_p, r, d, i_p = self.buffer[item + 1]
        return FullTransition(s, a, s_p, r, d, i_p)

    def __len__(self):
        if len(self.buffer) == 0:
            return 0
        _, _, _, done, _ = self.buffer[-1]
        """ if the final state is not done, then we are still writing """
        if not done:
            """ so we can't use the transition at the end yet"""
            return len(self.transitions) - 1
        return len(self.transitions)

    def append_buffer(self, buffer):
        for trajectory in buffer.trajectories:
            first = True
            for s, a, s_p, r, d, i in TrajectoryTransitions(buffer, trajectory):
                if first:
                    self.buffer.append((None, s, 0.0, False, {}))
                    first = False
                    self.eps_reward = 0
                    self.eps_len = 0
                self.transitions.append(len(self.buffer) - 1)
                self.buffer.append((a, s_p, r, d, i))
                self.eps_reward += r
                self.eps_len += 1
            self.trajectories.append((self.traj_start, len(self.buffer)))
            self.trajectory_info.append({'return': self.eps_reward, 'len': self.eps_len})


class ReplayBufferDataset:
    """
    Wraps the buffer to provide a convenient and efficient way to read transitions for batch collation

    Args:
        buffer: a replay buffer
        fields: a list of keys to retrieve from the buffer s: state a: action s_p: state prime, the resultant \
        state r: reward d: done
        info_keys: a single key, or list of keys to load from the transitions info dict
    """

    def __init__(self, buffer, fields=None, info_keys=None):
        self.buffer = buffer
        self.fields = fields if fields is not None else ['s', 'a', 's_p', 'r', 'd']
        if isinstance(info_keys, str):
            self.info_keys = [info_keys]
        self.info_keys = info_keys if info_keys is not None else []

    def __getitem__(self, item):
        item = self.buffer.transitions[item]
        _, s, _, _, _ = self.buffer.buffer[item]
        a, s_p, r, d, i = self.buffer.buffer[item + 1]
        map = {'s': s, 'a': a, 's_p': s_p, 'r': r, 'd': d}

        fields = []
        transition = []
        for field in self.fields:
            fields += [field]
            transition += [map[field]]

        for key in self.info_keys:
            fields += [key]
            transition += [i[key]]

        #Transition = namedtuple('Transition', fields) # local named tuple cannot be picked.

        return tuple(transition)

    def __len__(self):
        return len(self.buffer)


class TrajectoryTransitions:
    """
    Iterates over a trajectory in the buffer, from start to end, given a start:end tuple

    Args
        buffer: replay buffer
        trajectory_start_end_tuple: a tuple from buffer.trajectories

    eg: to iterate over the most recent trajectory

    .. code-block:: python

        trajectory = Transition(buffer, buffer.trajectories[-1])

    """

    def __init__(self, replay_buffer, trajectory_start_end_tuple):
        self.buffer = replay_buffer
        self.start = trajectory_start_end_tuple[0]
        self.end = trajectory_start_end_tuple[1]
        self.cursor = self.start

    def __next__(self):
        if self.cursor + 1 < self.end:
            _, s, _, _, _ = self.buffer.buffer[self.cursor]
            a, s_p, r, d, i = self.buffer.buffer[self.cursor + 1]
            self.cursor += 1
            return FullTransition(s, a, s_p, r, d, i)
        else:
            raise StopIteration

    def __iter__(self):
        return self


class TrajectoryTransitionsReverse:
    """
    Iterates in reverse over the provided trajectory
    replay_buffer: the replay buffer
    trajectory_start_end_tuple: a tuple with the start and end of the replay buffer
    """

    def __init__(self, replay_buffer, trajectory_start_end_tuple):
        self.buffer = replay_buffer
        self.start = trajectory_start_end_tuple[0]
        self.end = trajectory_start_end_tuple[1]
        self.cursor = self.end - 1

    def __next__(self):
        if self.cursor > self.start:
            _, s, _, _, _ = self.buffer.buffer[self.cursor - 1]
            a, s_p, r, d, i = self.buffer.buffer[self.cursor]
            self.cursor -= 1
            return FullTransition(s, a, s_p, r, d, i)
        else:
            raise StopIteration

    def __iter__(self):
        return self


def save(buffer, filepath):
    file = open(filepath, mode='wb')
    pickle.dump(buffer, file)
    file.close()


def load(filepath):
    file = open(filepath, 'rb')
    load_buff = pickle.load(file)
    file.close()
    return load_buff


def step_environment(env, policy, render=False, timing=False, **kwargs):
    """
    Transition generator, advances a single transition each iteration

    Args:
        env: gym environment to step
        policy: policy to use, policy(state) -> action
        render: calls env render function if True
        timing: prints timing info to stdout if True
        kwargs: will be passed to the policy
    """
    step_time, render_time, policy_time = [], [], []
    step_t, start_t, policy_t, render_t = 0, 0, 0, 0
    steps = 0
    done = True
    state = None

    while True:

        if timing:
            start_t = time.time()

        if done:
            state, state_info = env.reset(), {}
            if render:
                env.render()
        action = policy(state, **kwargs)
        if timing:
            policy_t = time.time()

        state_p, reward, done, info = env.step(action)

        if timing:
            step_t = time.time()

        if render:
            env.render()

        if timing:
            render_t = time.time()

        yield FullTransition(state, action, state_p, reward, done, info)

        state = state_p
        done = done
        steps += 1

        if timing:
            policy_time += [policy_t - start_t]
            step_time += [step_t - policy_t]
            render_time += [render_t - step_t]

            if steps % 100 == 0:
                print(f'policy_time {mean(policy_time)}, step_time {mean(step_time)}, render_time {mean(render_time)}')
                step_time, render_time, policy_time = [], [], []



