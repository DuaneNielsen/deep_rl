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
    """

    Args:
        dl: dataloader for sampling from replay buffer
        q: q function in form q(s) -> a where a is a value for each action
        target_q: target_q function for polyak update
        policy: policy(s) -> a_logits where a is the log probability of each action
        q_optim: optimizer for the q function
        policy_optim: optimizer for the policy
        discount: discount
        polyak: small number for polyak update, ie: 0.09
        alpha: soft entropy bonus
        device: cuda device
        precision: precision to train at

    """

    for s, a, s_p, r, d in dl:
        s = s.type(precision).to(device)
        a = a.to(device)
        s_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).unsqueeze(1)
        d = (1.0 * ~d.to(device)).unsqueeze(1)

        N = s.size(0)

        """ 1 step soft Q update """
        with torch.no_grad():
            a_p_dist_log = policy(s_p)
            v_sp = torch.sum(torch.exp(a_p_dist_log) * (q(s_p) - alpha * a_p_dist_log), dim=-1, keepdim=True)
            y = r + d * discount * v_sp

        ql = mse_loss(q(s)[torch.arange(N), a].unsqueeze(1), y)

        q_optim.zero_grad()
        ql.backward()
        q_optim.step()

        """ soft policy update """
        a_dist_log = policy(s)
        pl = torch.mean(torch.exp(a_dist_log) * (alpha * a_dist_log - q(s)))

        policy_optim.zero_grad()
        pl.backward()
        policy_optim.step()

        """ polyak update target_q """
        for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
            target_q_param.data.copy_(polyak * q_param.data + (1.0 - polyak) * target_q_param.data)

        break