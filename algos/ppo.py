import torch
from torch.utils.data import DataLoader
import buffer as bf


def ppo_loss(newlogprob, oldlogprob, advantage, clip=0.2):
    log_ratio = (newlogprob - oldlogprob)
    # clamp the log to stop infinities (85 is for 32 bit floats)
    log_ratio.clamp_(min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio)

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage.unsqueeze(1)
    full_step = ratio * advantage.unsqueeze(1)
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    # logger.info(f'mean advantage : {advantage.mean()}')
    # logger.info(f'mean newlog    : {newlogprob.mean()}')
    # logger.info(f'mean oldlob    : {oldlogprob.mean()}')
    # logger.info(f'mean log_ratio : {log_ratio.mean()}')
    # logger.info(f'mean ratio     : {ratio.mean()}')
    # logger.info(f'mean clip ratio: {clipped_ratio.mean()}')
    # logger.info(f'mean clip step : {clipped_step.mean()}')

    return - min_step.mean()


def train(buffer, policy_net, optim, clip=0.2):
    ds = bf.ReplayBufferDataset(buffer, fields=('s', 'a'), info_keys='advantage')
    dl = DataLoader(ds, batch_size=10000, num_workers=0)

    for state, action, advantage in dl:

        optim.zero_grad()
        new_logprob = policy_net(state).logprob(action)
        old_logprob = policy_net(state, old=True).logprob(action)
        loss = ppo_loss(new_logprob, old_logprob, advantage, clip=clip)
        loss.backward()
        optim.step()