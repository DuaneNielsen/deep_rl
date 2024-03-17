from gym.envs.registration import register
from gymnasium.envs.registration import register as gymnasium_register

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

register(
    id='MnistBandit-v1',
    entry_point='env.debug:MnistBandit',
    max_episode_steps=1,
    reward_threshold=1.0,
)

register(
    id='MnistDelayedBandit-v1',
    entry_point='env.debug:MnistDelayedBandit',
    max_episode_steps=50,
    reward_threshold=1.0,
)

register(
    id='MnistBanditEasy-v1',
    entry_point='env.debug:MnistBandit',
    max_episode_steps=1,
    reward_threshold=1.0,
    kwargs={'easy': True}
)

register(
    id='MnistDelayedBanditEasy-v1',
    entry_point='env.debug:MnistDelayedBandit',
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={'easy': True}
)

register(
    id='MnistTargetEasy-v1',
    entry_point='env.debug:MnistTargetGrid',
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={'initial_state': 3, 'n_states': 7, 'easy': True}
)

register(
    id='MnistTarget-v1',
    entry_point='env.debug:MnistTargetGrid',
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={'initial_state': 3, 'n_states': 7, 'easy': False}
)

gymnasium_register(
    id='TugOfWar-v1',
    entry_point='env.gymnasium_debug:TugOfWar',
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={}
)

gymnasium_register(
    id='IteratedRockPaperScissors-v1',
    entry_point='env.gymnasium_debug:IteratedRockPaperScissors',
    max_episode_steps=50,
    reward_threshold=1.0,
    kwargs={"max_iterations": 3}
)
