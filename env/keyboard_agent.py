#!/usr/bin/env python
import sys, gym, time
import numpy as np
import env.wrappers as wrappers
import matplotlib.pyplot as plt

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

#env = gym.make('BreakoutDeterministic-v4' if len(sys.argv)<2 else sys.argv[1])


class Gradient(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev = None

    def reset(self):
        state = self.env.reset()
        self.prev = state
        grad = state - self.prev
        return np.concatenate((state, grad), axis=2)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        grad = state - self.prev
        self.prev = state

        return np.concatenate((state, grad), axis=2), reward, done, info


def make_env():
    env = gym.make('BreakoutDeterministic-v4')
    env = wrappers.EpisodicLifeEnv(env)
    env = wrappers.NoopResetEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = wrappers.FireResetEnv(env)
    env = wrappers.ClipState2D(env, 0, 24, 210 - 24, 160)
    env = wrappers.WarpFrame(env)
    env = Gradient(env)
    env = wrappers.ClipRewardEnv(env)
    env = wrappers.PenalizeOneIfDone(env)

    # env = RemapActions(env, remap=np.array([1, 2, 3]))

    # env = LiveMonitor(env)
    return env

plt.ion()
fig = plt.figure()
spec = plt.GridSpec(nrows=1, ncols=2, figure=fig)

ax_state = fig.add_subplot(fig.add_subplot(spec[0, 0]))
ax_grad = fig.add_subplot(fig.add_subplot(spec[0, 1]))

env = make_env()

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

_keymap = {ord(' '): 1,
       ord('a'): 3,
       ord('d'): 2 }


def keymap(key):
    if key in _keymap:
        return _keymap[key]
    else:
        return 0


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = keymap(key)
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = keymap(key)
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def plot(obser):
    ax_state.clear()
    ax_grad.clear()
    ax_state.imshow(obser[:, :, 0])
    ax_grad.imshow(obser[:, :, 1])
    fig.canvas.draw()


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    print('RESET')
    plot(obser)
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        plot(obser)
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            print('PAUSED')
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break