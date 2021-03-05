import time
import torch


def render_env(env, render, delay):
    if render:
        env.render(render)
        time.sleep(delay)


def episode(env, policy, render=False, delay=0.01, **kwargs):
    """
    Runs one episode using the provided policy on the environment
    :param env: gym environment to generate an episode for
    :param policy: takes state as input, must output an action runnable on the environment
    :param render: if True will call environments render function
    :param delay: rendering delay
    :param kwargs: kwargs will be passed to policy, environment step, and observers
    """
    with torch.no_grad():
        state, reward, done, info = env.reset(), 0.0, False, {}
        action = policy(state, **kwargs)
        render_env(env, render, delay)
        while not done:
            state, reward, done, info = env.step(action)
            action = policy(state)
            render_env(env, render, delay)


def step_environment(env, policy, render=False, **kwargs):
    """
    Transition generator, advances a single transition each iteration
    :param env: gym environment to step
    :param policy: policy use
    :param render: calls env render function if True
    :param kwargs: will be passed to the policy, and environment
    """
    done = True
    state = None

    while True:
        if done:
            state, state_info = env.reset(), {}
            if render:
                env.render()
        with torch.no_grad():
            action = policy(state, **kwargs)
        state_p, reward, done, info = env.step(action)
        if render:
            env.render()

        yield state, action, state_p, reward, done, info

        state = state_p
        done = done