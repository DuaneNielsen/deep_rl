import torch
from torch.nn.functional import mse_loss
from logger import logger


def stats_dict(name, tensor):
    st = {}
    st[name + ' Mean'] = tensor.mean().item()
    st[name + ' Std'] = tensor.std().item()
    st[name + ' Max'] = tensor.max().item()
    st[name + ' Min'] = tensor.min().item()
    st[name + ' histogram'] = tensor.detach().cpu().numpy()
    return st


def train_continuous(dl, q, target_q, policy, q_optim, policy_optim,
                     sample_actions=8, amin=-1.0, amax=1.0,
                     discount=0.99, polyak=0.095, q_update_ratio=2, policy_alpha=0.2, cql_alpha=1.0,
                     device='cpu', precision=torch.float32, log=False
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

        with torch.no_grad():
            a_dist = policy(s)
            a_p_dist = policy(s_p)
            a_p = a_p_dist.rsample()
            y = r + d * discount * target_q(s_p, a_p)

            a_sample = []
            a_sample += [torch.empty(sample_actions, N, A, device=device).uniform_(amin, amax)]
            a_sample += [a_dist.rsample((sample_actions, ))]
            a_sample += [a_p_dist.rsample((sample_actions, ))]
            a_sample = torch.stack(a_sample, dim=0).reshape(sample_actions * 3 * N, A)
            s_sample = s.view(1, N, S)[torch.zeros(sample_actions * 3, dtype=torch.long), :, :].reshape(sample_actions * 3 * N, S)

        # the reference implementation includes q_replay in the logsumexp term
        # and does not detach gradients from the in-distribution action term
        q_sample = q(s_sample, a_sample).reshape(sample_actions * 3, N, 1, -1)
        q_replay = q(s, a)
        cql_loss = torch.logsumexp(q_sample, dim=0) - q_replay

        td_loss = (q_replay - y) ** 2 / 2
        qloss = torch.mean(td_loss + cql_alpha * cql_loss)

        q_optim.zero_grad()
        qloss.backward()
        q_optim.step()

        if q_update % q_update_ratio > 0:
            q_update += 1
            continue

        a_dist = policy(s)
        a_ = a_dist.rsample()
        min_q, _ = torch.min(q(s, a_), dim=2)
        log_pi = a_dist.log_prob(a_).sum(1, keepdim=True)
        policy_loss = - torch.mean(min_q - policy_alpha * log_pi)

        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()
        q_optim.zero_grad()

        for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
            target_q_param.data.copy_(polyak * q_param.data + (1.0 - polyak) * target_q_param.data)

        if log:

            logger.log['trainer-Policy Loss'] = policy_loss.item()
            logger.log.update(stats_dict('trainer-Log Pis', log_pi))
            logger.log.update(stats_dict('trainer-Policy mu', a_dist.mu))
            logger.log.update(stats_dict('trainer-Policy log std', torch.log(a_dist.scale)))

            logger.log['trainer-Q loss'] = qloss.item()
            logger.log.update(stats_dict('trainer-Q1 Predictions', q_replay[..., 0]))
            logger.log.update(stats_dict('trainer-Q2 Predictions', q_replay[..., 1]))
            logger.log.update(stats_dict('trainer-Q Targets', y))
            logger.log['trainer-QF1 Loss'] = td_loss[..., 0].mean().item()
            logger.log['trainer-min QF1 Loss'] = cql_loss[..., 0].mean().item()
            logger.log['trainer-QF2 Loss'] = td_loss[..., 1].mean().item()
            logger.log['trainer-min QF2 Loss'] = cql_loss[..., 1].mean().item()
            logger.log['trainer-Std QF1 values'] = q_sample[..., 0].std().item()
            logger.log['trainer-Std QF2 values'] = q_sample[..., 1].std().item()
            logger.log.update(stats_dict('trainer-QF1 in-distribution values', q_sample[sample_actions:sample_actions * 2, ..., 0]))
            logger.log.update(stats_dict('trainer-QF2 in-distribution values', q_sample[sample_actions:sample_actions * 2, ..., 1]))
            logger.log.update(stats_dict('trainer-QF1 random values', q_sample[0:sample_actions, ..., 0]))
            logger.log.update(stats_dict('trainer-QF2 random values', q_sample[0:sample_actions, ..., 1]))
            logger.log.update(stats_dict('trainer-QF1 next_actions values', q_sample[sample_actions*2:sample_actions*3, ..., 0]))
            logger.log.update(stats_dict('trainer-QF2 next_actions values', q_sample[sample_actions*2:sample_actions*3, ..., 1]))

            logger.log.update(stats_dict('trainer-actions', a))
            logger.log.update(stats_dict('trainer-rewards', r))

        break


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
