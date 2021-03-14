from driver import episode
from capture import PngCapture
from env.wrappers import ClipState2D, Greyscale
import gym
from random import randint
from pathlib import Path
import torch.utils.data
import torchvision.datasets
from matplotlib import pyplot as plt


pongdir = 'data/Pong_1'

if __name__ == '__main__':

    def policy(state):
        """ a random policy to generate actions in Pong"""
        return randint(2, 3)


    if not Path(pongdir).exists():
        env = gym.make('Pong-v0')
        env = ClipState2D(env, 0, 24, 210-24, 160)
        env = PngCapture(env, pongdir + '/screens')
        episode(env, policy)
        episode(env, policy)
        episode(env, policy)

    train_dataset = torchvision.datasets.ImageFolder(
        root=pongdir,
        transform=torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True
    )

    plt.ion()
    fig = plt.figure(figsize=(8, 4))
    axe = [fig.add_subplot(2, 4, i+1) for i in range(8)]

    for batch, _ in train_loader:
        for i, image in enumerate(batch):
            axe[i].imshow(image.permute(1, 2, 0))
        plt.pause(0.4)