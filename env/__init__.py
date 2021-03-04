from gym.envs.registration import register

register(
    id='CartPoleContinuous-v1',
    entry_point='env.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=200.0,
)

register(
    id='Bandit-v1',
    entry_point='env.debug:Bandit',
    max_episode_steps=1,
    reward_threshold=1.0,
)

register(
    id='DelayedBandit-v1',
    entry_point='env.debug:DelayedBandit',
    max_episode_steps=50,
    reward_threshold=1.0,
)


