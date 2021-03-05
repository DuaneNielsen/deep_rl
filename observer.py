from collections import OrderedDict

import gym


class EnvObserver:
    def reset(self, state):
        """ called when environment reset"""
        pass

    def step(self, action, state, reward, done, info, **kwargs):
        """ called each environment step """
        pass


class StateCapture(EnvObserver):
    def __init__(self):
        self.trajectories = []
        self.traj = []
        self.index = []
        self.cursor = 0

    def reset(self, state):
        self.traj.append(state)
        self.index.append(self.cursor)
        self.cursor += 1

    def step(self, action, state, reward, done, info, **kwargs):
        self.traj.append(state)
        self.index.append(self.cursor)
        self.cursor += 1

        if done:
            self.done()

    def done(self):
        self.trajectories += [self.traj]
        self.traj = []


class StepFilter:
    """
    Step filters are used to preprocess steps before handing them to observers

    if you want to pre-process environment observations before passing to policy, use a gym.Wrapper
    """

    def __call__(self, action, state, reward, done, info, **kwargs):
        return action, state, reward, done, info, kwargs


class RewardFilter(StepFilter):
    def __init__(self, state_prepro, R, device):
        self.state_prepro = state_prepro
        self.R = R
        self.device = device

    def __call__(self, action, state, reward, done, info, **kwargs):
        r = self.R(self.state_prepro(state, self.device))
        kwargs['model_reward'] = r.item()
        return action, state, reward, done, info, kwargs


class SubjectWrapper(gym.Wrapper):
    """
    gym wrapper with pluggable observers

    to attach an observer implement EnvObserver interface and use attach()

    filters to process the steps are supported, and data enrichment is possible
    by adding to the kwargs dict
    """

    def __init__(self, env, seed=None, **kwargs):
        gym.Wrapper.__init__(self, env)
        self.kwargs = kwargs
        self.env = env
        if seed is not None:
            env.seed(seed)
        self.observers = OrderedDict()
        self.step_filters = OrderedDict()

    def attach_observer(self, name, observer):
        self.observers[name] = observer

    def detach_observer(self, name):
        del self.observers[name]

    def observer_reset(self, state):
        for name, observer in self.observers.items():
            observer.reset(state)

    def append_step_filter(self, name, filter):
        self.step_filters[name] = filter

    def observe_step(self, action, state, reward, done, info, **kwargs):
        for name, filter in self.step_filters.items():
            action, state, reward, done, info, kwargs = filter(action, state, reward, done, info, **kwargs)

        for name, observer in self.observers.items():
            observer.step(action, state, reward, done, info, **kwargs)

    def reset(self):
        state = self.env.reset()
        self.observer_reset(state)
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.observe_step(action, state, reward, done, info)
        return state, reward, done, info