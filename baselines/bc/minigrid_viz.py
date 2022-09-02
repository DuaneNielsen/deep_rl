import gym
import gym_minigrid
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

H, W = 8*2 +1, 8*2+1


# convert co-ordinates to index
def idx(x, y):
    return x * H + y


# return co-ordinates for N, S, E, W triangle with respect to the center triangle
def north(x, y):
    return (x, y), (x + 1, y + 1), (x - 1, y + 1)


def east(x, y):
    return (x, y), (x + 1, y - 1), (x + 1, y + 1)


def south(x, y):
    return (x, y), (x - 1, y - 1), (x + 1, y - 1)


def west(x, y):
    return (x, y), (x - 1, y + 1), (x - 1, y - 1)


if __name__ == '__main__':
    from minigrid import PolicyNet, prepro_obs
    from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper
    from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
    from env.wrappers import ApplyFunc
    from argparse import ArgumentParser
    from torch.optim import Adam
    import torch
    import torch_utils

    matplotlib.use('TkAgg')

    parser = ArgumentParser()
    parser.add_argument('--run_dir', type=str)
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-8x8-v0')
    parser.add_argument('--max_steps', type=int, default=4)
    args = parser.parse_args()

    """ environment """
    env = gym.make(args.env_name, new_step_api=True)
    env = FullyObsWrapper(env)  # Use full observations
    env = ImgObsWrapper(env)
    env = ApplyFunc(env, prepro_obs, gym.spaces.Space(shape=(3, 8, 8), dtype=np.float32))
    env.unwrapped.max_steps = args.max_steps

    """ policy """
    policy_net = PolicyNet(linear_in_dims=16, actions=3)
    optim = Adam(policy_net.parameters())
    torch_utils.load_checkpoint(args.run_dir, prefix='best', policy_net=policy_net, optim=optim)


    def policy_net_eval(s):
        """ wrap the policy for inference"""
        with torch.no_grad():
            s = torch.from_numpy(s).to(policy_net.dummy_param.dtype)
            a = policy_net(s.unsqueeze(0))
            return torch.argmax(a).item()

    # compute all the center co-ordinates in the grid
    c_x, c_y = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')

    # enumerate all agent positions and orientations in the MDP, including unreachable ones
    env = env.unwrapped
    env.reset()
    full_grid = env.grid.encode()
    triangles = []
    colors = []

    for i, j in zip(c_x.flatten(), c_y.flatten()):
        c_x, c_y = i * 2 + 1, j * 2 + 1
        for agent_dir in range(4):
            obs = full_grid.copy()
            obs[i, j] = np.array(
                [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], agent_dir]
            )
            observation = prepro_obs(obs)
            action = policy_net_eval(observation)

            if agent_dir == 0:
                triangles += [east(c_x, c_y)]
            if agent_dir == 1:
                triangles += [south(c_x, c_y)]
            if agent_dir == 2:
                triangles += [west(c_x, c_y)]
            if agent_dir == 3:
                triangles += [north(c_x, c_y)]

            colors += [~(action == 2)]

    # convert triangles from xy to vertex index
    triangles = [[idx(*xy) for xy in tri] for tri in triangles]

    # vertices at grid co-ordinates in increments of 0.5
    x, y = np.meshgrid(np.linspace(0, H//2, H), np.linspace(0, W//2, W), indexing='ij')

    fig = plt.figure()
    ax = plt.subplot()
    ax.tripcolor(x.flatten(), y.flatten(), triangles, colors, cmap='summer')
    plt.gca().invert_yaxis()
    plt.show()


