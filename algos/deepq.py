import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import algos.utils


def train(buffer, q_net, optim, batch_size, discount, device='cpu', dtype=torch.float):

    sampler = SubsetRandomSampler(random.sample(range(len(buffer)), batch_size))

    dl = DataLoader(buffer, batch_size=batch_size, sampler=sampler)

    for s, i, a, s_p, r, d, i_p in dl:
        s, s_p, r, d = algos.utils.to(s, s_p, r, d, device=device, dtype=dtype)
        a = a.to(device)
        N = s.shape[0]

        optim.zero_grad()
        v0 = q_net(s)[torch.arange(N), a].unsqueeze(1)
        with torch.no_grad():
            v1, action = torch.max(q_net(s_p), dim=1, keepdim=True)
            v1[d] = 0.0
            v1 = r + v1 * discount
        loss = torch.mean((v1 - v0) ** 2)

        loss.backward()
        optim.step()


