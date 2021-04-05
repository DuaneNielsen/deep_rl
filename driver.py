import time
from statistics import mean
import torch


def _render_env(env, render, delay):
    if render:
        env.render()
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
        with torch.no_grad():
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

        yield state, action, state_p, reward, done, info

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


