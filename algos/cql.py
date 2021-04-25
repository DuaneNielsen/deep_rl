import torch
from torch.nn.functional import mse_loss


def train_continuous(dl, q, target_q, policy, q_optim, policy_optim,
                     sample_actions=8, amin=-1.0, amax=1.0,
                     discount=0.99, polyak=0.095, q_update_ratio=2, policy_alpha=0.2, cql_alpha=1.0,
                     device='cpu', precision=torch.float32
                     ):
    q_update = 1

    for s, a, s_p, r, d in dl:
        N = s.shape[0]
        S = s.shape[1]
        A = a.shape[1]
        s = s.type(precision).to(device)
        a = a.to(device)
        s_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).reshape(N, 1, 1)
        d = (1.0 * ~d.to(device)).reshape(N, 1, 1)

        a_dist = policy(s)

        with torch.no_grad():
            a_p_dist = policy(s_p)
            a_p = a_p_dist.rsample()
            y = r + d * discount * target_q(s_p, a_p)

            a_sample = []
            a_sample += [torch.empty(sample_actions, N, A, device=device).uniform_(amin, amax)]
            a_sample += [a_dist.rsample((sample_actions, ))]
            a_sample += [a_p_dist.rsample((sample_actions, ))]
            a_sample = torch.stack(a_sample, dim=0).reshape(sample_actions * 3 * N, A)
            s_sample = s.view(1, N, S)[torch.zeros(sample_actions * 3, dtype=torch.long), :, :].reshape(sample_actions * 3 * N, S)

        q_sample = q(s_sample, a_sample).reshape(sample_actions * 3, N, 1, -1)
        q_replay = q(s, a)
        cql_loss = torch.logsumexp(q_sample, dim=0) - q_replay.detach()

        td_loss = (q_replay - y) ** 2 / 2
        qloss = torch.mean(td_loss + cql_alpha * cql_loss)

        q_optim.zero_grad()
        qloss.backward()
        q_optim.step()

        if q_update % q_update_ratio > 0:
            q_update += 1
            continue

        a_ = a_dist.rsample()
        min_q, _ = torch.min(q(s, a_), dim=2)  # investigate why running gradients through here is critical
        pl = - torch.mean(min_q - policy_alpha * a_dist.log_prob(a_).sum(1, keepdim=True))

        policy_optim.zero_grad()
        pl.backward()
        policy_optim.step()
        q_optim.zero_grad()

        for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
            target_q_param.data.copy_(polyak * q_param.data + (1.0 - polyak) * target_q_param.data)

        break

    return {
        'cql_loss': torch.mean(cql_loss).item(),
        'q_loss': qloss.item(),
        'td_loss': torch.mean(td_loss).item(),
        'policy_loss': pl.item()
    }


def train_discrete(dl, q, target_q, policy, q_optim, policy_optim,
                   discount=0.99, polyak=0.095, q_update_ratio=2, policy_alpha=0.2, cql_alpha=1.0,
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

    q_update = 1

    for s, a, s_p, r, d in dl:
        s = s.type(precision).to(device)
        a = a.to(device)
        s_p = s_p.type(precision).to(device)
        r = r.type(precision).to(device).unsqueeze(1)
        d = (1.0 * ~d.to(device)).unsqueeze(1)
        N = s.size(0)
        A = policy.actions

        """ 1 step soft Q update """

        with torch.no_grad():
            a_p_dist = policy(s_p)
            v_sp = torch.sum(a_p_dist * q(s_p), dim=-1, keepdim=True)
            y = r + d * discount * v_sp

        v = q(s)
        q_pred = v[torch.arange(N), a].unsqueeze(1)
        cql = torch.logsumexp(v, dim=1, keepdim=True) - q_pred.detach()

        ql = mse_loss(q_pred, y) + cql_alpha * cql.mean()

        q_optim.zero_grad()
        ql.backward()
        q_optim.step()

        if q_update % q_update_ratio > 0:
            q_update += 1
            continue

        """ soft policy update """
        a_dist = policy(s)
        pl = torch.mean(a_dist * (policy_alpha * torch.log(a_dist) - q(s)))

        policy_optim.zero_grad()
        pl.backward()
        policy_optim.step()
        q_optim.zero_grad(set_to_none=True)

        """ polyak update target_q """
        for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
            target_q_param.data.copy_(polyak * q_param.data + (1.0 - polyak) * target_q_param.data)

        break
