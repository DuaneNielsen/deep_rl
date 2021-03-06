from collections import namedtuple
import gym


class Enricher:
    """
    Base class used to enrich data collected during run
    will be called after buffer operations are complete
    multiple enrichers will be called in order they were attached
    """

    def reset(self, buffer, state, **kwargs):
        pass

    def step(self, buffer, action, state, reward, done, info, **kwargs):
        pass

    def enrich(self, fields, transition, i, i_p):
        return fields, transition, i, i_p


class Returns(Enricher):
    """
    An enricher that calculates total returns
    Returns are added to the info field
    for transition (s, i, a, s_p, r, d, i_p), return = transition.i['g']
    """

    def __init__(self, key='g'):
        self.key = key

    def step(self, buffer, action, state, reward, done, info, **kwargs):
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

    def step(self, buffer, action, state, reward, done, info, **kwargs):
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

    def enrich(self, enricher):
        self._enrich.append(enricher)

    def clear(self):
        """ clears the buffer """
        self.buffer = []
        self.trajectories = []
        self.transitions = []
        self.traj_start = 0

    def reset(self):
        state = self.env.reset()
        self.buffer.append((None, state, 0.0, False, {}))
        self.transitions.append(len(self.buffer) - 1)
        self.eps_reward = 0
        self.eps_len = 0

        for enricher in self._enrich:
            enricher.reset(self, state)
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.buffer.append((action, state, reward, done, info))
        self.eps_reward += reward
        self.eps_len += 1

        if done:
            """ terminal state, trajectory is complete """
            self.trajectories.append((self.traj_start, len(self.buffer)))
            self.traj_start = len(self.buffer)
            self.trajectory_info.append({'R': self.eps_reward, 'len': self.eps_len})
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


# ['s', 'a'], ['advantage']

class ReplayBufferDataset:
    """
    ReplayBufferDataset

    wraps the buffer to provide a convenient and efficient way to read transitions for batch collation

    buffer: a replay buffer
    fields: a list of keys to retrieve from the buffer,
        Key:
            s: state
            a: action
            s_p: state prime, the resultant state
            r: reward
            d: done
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

    eg: to iterate over the most recent trajectory

    ```
    trajectory = Transition(buffer, buffer.trajectories[-1])
    ```

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

    ```
    env, buffer = buffer.wrap(env)
    ```
    """
    buffer = ReplayBuffer(env)
    return buffer, buffer