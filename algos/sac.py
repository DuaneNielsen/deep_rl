import torch
from torch.nn.functional import mse_loss


def train(dl, q, target_q, policy, q_optim, policy_optim,
          discount=0.99, polyak=0.095, alpha=0.2,
          device='cpu', precision=torch.float32):

    for s, a, s_p, r, d in dl:
        s = s.type(precision).to(device)
        a = a.to(device)
        s_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).unsqueeze(1)
        d = (1.0 * ~d.to(device)).unsqueeze(1)

        with torch.no_grad():
            a_p_dist = policy(s_p)
            a_p = a_p_dist.rsample()
            y = r + d * discount * (target_q(s_p, a_p) - alpha * a_p_dist.log_prob(a_p).sum(1, keepdim=True))

        ql = mse_loss(q(s, a), y)

        q_optim.zero_grad()
        ql.backward()
        q_optim.step()

        a_dist = policy(s)
        a_ = a_dist.rsample()
        pl = - torch.mean(q(s, a_) - alpha * a_dist.log_prob(a_).sum(1, keepdim=True))

        policy_optim.zero_grad()
        pl.backward()
        policy_optim.step()

        for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
            target_q_param.data.copy_(polyak * q_param.data + (1.0 - polyak) * target_q_param.data)

        break


def train_discrete(dl, q, target_q, policy, q_optim, policy_optim,
          discount=0.99, polyak=0.095, alpha=0.2,
          device='cpu', precision=torch.float32):

    for s, a, s_p, r, d in dl:
        s = s.type(precision).to(device)
        a = a.to(device)
        s_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).unsqueeze(1)
        d = (1.0 * ~d.to(device)).unsqueeze(1)

        N = s.size(0)

        with torch.no_grad():
            a_p_dist = policy(s_p)
            a_p = a_p_dist.rsample()
            y = r + d * discount * (torch.sum(target_q(s_p) * a_p, dim=1, keepdim=True) - alpha * a_p_dist.log_prob(a_p).unsqueeze(1))

        ql = mse_loss(q(s)[torch.arange(N), a].unsqueeze(1), y)

        q_optim.zero_grad()
        ql.backward()
        q_optim.step()

        a_dist = policy(s)
        a_ = a_dist.rsample()
        pl = - torch.mean(torch.sum(q(s) * a_, dim=1, keepdim=True) - alpha * a_dist.log_prob(a_))

        policy_optim.zero_grad()
        pl.backward()
        policy_optim.step()

        for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
            target_q_param.data.copy_(polyak * q_param.data + (1.0 - polyak) * target_q_param.data)

        break