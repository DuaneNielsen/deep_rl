import time
import torch


def _render_env(env, render, delay):
    if render:
        env.render(render)
        time.sleep(delay)


def episode(env, policy, render=False, delay=0.01, **kwargs):
    """
    Runs one episode using the provided policy on the environment

    Args:
        env: gym environment to generate an episode for
        policy: policy(state) -> action takes state as input, must output an action runnable on the environment
        render: if True will call environments render function
        delay: rendering delay
        kwargs: kwargs will be passed to policy
    """
    with torch.no_grad():
        state, reward, done, info = env.reset(), 0.0, False, {}
        action = policy(state, **kwargs)
        _render_env(env, render, delay)
        while not done:
            state, reward, done, info = env.step(action)
            action = policy(state)
            _render_env(env, render, delay)


def step_environment(env, policy, render=False, **kwargs):
    """
    Transition generator, advances a single transition each iteration

    Args:
        env: gym environment to step
        policy: policy to use, policy(state) -> action
        render: calls env render function if True
        kwargs: will be passed to the policy
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

