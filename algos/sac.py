import torch
from torch.nn.functional import mse_loss


def train(dl, q, target_q, policy, q_optim, policy_optim,
          discount=0.99, polyak=0.095, q_update_ratio=2, alpha=0.2,
          device='cpu', precision=torch.float32):

    q_update = 1

    for s, a, s_p, r, d in dl:
        N = s.shape[0]
        s = s.type(precision).to(device)
        a = a.to(device)
        s_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).reshape(N, 1, 1)
        d = (1.0 * ~d.to(device)).reshape(N, 1, 1)

        with torch.no_grad():
            a_p_dist = policy(s_p)
            a_p = a_p_dist.rsample()
            y = r + d * discount * (target_q(s_p, a_p) - alpha * a_p_dist.log_prob(a_p).sum(1, keepdim=True).unsqueeze(2))

        ql = mse_loss(q(s, a), y)

        q_optim.zero_grad()
        ql.backward()
        q_optim.step()

        if q_update % q_update_ratio > 0:
            q_update += 1
            continue

        a_dist = policy(s)
        a_ = a_dist.rsample()
        min_q, _ = torch.min(q(s, a_), dim=2)
        pl = - torch.mean(min_q - alpha * a_dist.log_prob(a_).sum(1, keepdim=True))

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
        q_optim.zero_grad()

        """ 1 step soft Q update """
        with torch.no_grad():
            a_p_dist = policy(s_p)
            v_sp = torch.sum(a_p_dist * (q(s_p) - alpha * torch.log(a_p_dist)), dim=-1, keepdim=True)
            y = r + d * discount * v_sp

        ql = mse_loss(q(s)[torch.arange(N), a].unsqueeze(1), y)

        ql.backward()
        # q_total_norm = 0
        # for p in q.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     q_total_norm += param_norm.item() ** 2
        # q_total_norm = q_total_norm ** (1. / 2)
        q_optim.step()

        policy_optim.zero_grad()

        """ soft policy update """
        a_dist = policy(s)
        #print(a_dist_log.T)
        pl = torch.mean(a_dist * (alpha * torch.log(a_dist) - q(s)))
        #a_dist_probs = torch.exp(a_dist_log)
        #print(pl.item(), torch.sum(alpha * a_dist_log * a_dist_probs).item()/N, -torch.sum(q(s)[torch.arange(N), a].unsqueeze(1) * a_dist_probs).item()/N, torch.mean(y).item())
        pl.backward()
        #torch.nn.utils.clip_grad_norm_(policy.parameters(), 1e-5)
        # total_norm = 0
        #
        # for p in policy.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # print(total_norm, q_total_norm)
        policy_optim.step()

        q_optim.zero_grad(set_to_none=True)
        """ polyak update target_q """
        for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
            target_q_param.data.copy_(polyak * q_param.data + (1.0 - polyak) * target_q_param.data)

        break