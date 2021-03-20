from collections import namedtuple
import gym

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


class Enricher:
    """
    Base class used to enrich data collected during run
    will be called after buffer operations are complete
    multiple enrichers will be called in order they were attached
    """

    def reset(self, buffer, state):
        """ Reset will be called when environment is reset

        Args:
            buffer: replay buffer
            state: the state returned by the environment
        """
        pass

    def step(self, buffer, action, state, reward, done, info):
        """
        step will be called when the environment step function is run

        Args:
            buffer: the replay buffer
            action: action taken
            state: resultant state after taking action
            reward: resultant reward
            done: true if this is last step in trajectory
            info: info dict returned by environment store output in info if enriching the step output\
            recording trajectory information can be stored in buffer.trajectory_info
        """
        pass


class Returns(Enricher):
    """
    An enricher that calculates total returns

    Args:
        key: key to use to add the returns, default is 'g'

    use the key value to retrieve the returns from ReplayBufferDataset

    .. code-block :: python

        buffer.enrich(Returns(key='returns'))
        ds = ReplayBufferDataset(buffer, info_keys='returns')

        s, a, s_p, r, d, R = next(ds)

    """
    def __init__(self, key='g'):
        self.key = key

    def step(self, buffer, action, state, reward, done, info):
        """ computes return """
        if done:
            # terminal state returns are always 0
            g = 0
            for s, a, s_p, r, d, i in TrajectoryTransitionsReverse(buffer, buffer.trajectories[-1]):
                g += r
                i[self.key] = g


class DiscountedReturns(Enricher):
    """
    Enriches the transitions with discounted returns
    Returns are added to the info field
    for transition (s, i, a, s_p, r, d, i_p), return = transition.i['g']
    """

    def __init__(self, key='g', discount=0.95):
        self.key = key
        self.discount = discount

    def step(self, buffer, action, state, reward, done, info):
        """ computes discounted return """
        if done:
            # terminal state returns are always 0
            g = 0.0
            # get the last trajectory and reverse iterate over transitions
            for s, a, s_p, r, d, i in TrajectoryTransitionsReverse(buffer, buffer.trajectories[-1]):
                g = r + g * self.discount
                i[self.key] = g


class ReplayBuffer(gym.Wrapper):
    def __init__(self, env):
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
        super().__init__(env)
        self.buffer = []
        self.trajectories = []
        self.trajectory_info = []
        self.transitions = []
        self.traj_start = 0
        self._enrich = []
        self.eps_reward = 0
        self.eps_len = 0
        self.record = True

    def enrich(self, enricher):
        """
        Append an enricher to enrich the data collected by the buffer
        Args:
            enricher: object that implements buffer.Enricher

        For example, to enrich the buffer with discounted returns...

        .. code-block:: python

            buffer.enrich(buffer.DiscountedReturns(discount=0.9))

        """
        self._enrich.append(enricher)

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

    def reset(self):
        """  Wraps the gym reset method """
        state = self.env.reset()

        if self.record:
            self.buffer.append((None, state, 0.0, False, {}))
            self.transitions.append(len(self.buffer) - 1)
            self.eps_reward = 0
            self.eps_len = 0

            for enricher in self._enrich:
                enricher.reset(self, state)
        return state

    def step(self, action):
        """  Wraps the gym step method """
        state, reward, done, info = self.env.step(action)

        if self.record:
            self.buffer.append((action, state, reward, done, info))
            self.eps_reward += reward
            self.eps_len += 1

            if done:
                """ terminal state, trajectory is complete """
                self.trajectories.append((self.traj_start, len(self.buffer)))
                self.traj_start = len(self.buffer)
                self.trajectory_info.append({'return': self.eps_reward, 'len': self.eps_len})
            else:
                """ if not terminal, then by definition, this will be a transition """
                self.transitions.append(len(self.buffer) - 1)

            for enricher in self._enrich:
                enricher.step(self, action, state, reward, done, info)

        return state, reward, done, info

    def __getitem__(self, item):
        item = self.transitions[item]
        _, s, _, _, i = self.buffer[item]
        a, s_p, r, d, i_p = self.buffer[item + 1]
        Transition = namedtuple('Transition', ['s', 'a', 's_p', 'r', 'd'])
        return Transition(s, a, s_p, r, d)

    def __len__(self):
        if len(self.buffer) == 0:
            return 0
        _, _, _, done, _ = self.buffer[-1]
        """ if the final state is not done, then we are still writing """
        if not done:
            """ so we can't use the transition at the end yet"""
            return len(self.transitions) - 1
        return len(self.transitions)




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

        Transition = namedtuple('Transition', fields)

        return Transition(*transition)

    def __len__(self):
        return len(self.buffer)


FullTransition = namedtuple('FullTransition', ['s', 'a', 's_p', 'r', 'd', 'i'])


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


def wrap(env):
    """
    convenience method for wrapping a gym environment

    .. code-block:: python

        env, buffer = buffer.wrap(env)
    """
    buffer = ReplayBuffer(env)
    return buffer, buffer
