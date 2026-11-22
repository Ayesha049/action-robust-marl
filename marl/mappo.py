"""Minimal MAPPO for VMAS tasks (Phase 2 Transport base).

Policy: tanh-squashed Gaussian. The mean network is a marl.nets.Actor (tanh(mu(obs))), so the
deterministic policy saved as actor_<i>.pt is exactly the MADDPG checkpoint format, and
load_team / eval_clean / the Stage 0 fault harness use it unchanged. log_std is a
state-independent parameter per action dimension.

Ratio: actions are sampled as u ~ N(mu, std), a = tanh(u), and u is stored. The tanh Jacobian
term of log pi(a) depends on u only, so it cancels in the PPO ratio, which therefore uses the
Gaussian log-density of u.

Critic: one centralised V(concatenated observations of all agents), trained on the shared team
reward (VMAS transport gives every agent the same shaped reward). Actor parameters are shared
across agents by default (homogeneous agents, as in the VMAS paper's MAPPO); --no-share gives
one actor per agent.

Rollouts: VMAS's own vectorised envs (n_envs in one batch). Each iteration resets every env and
runs one full fixed-horizon episode, so a batch is n_envs x horizon frames. The last state is
a truncation, not a terminal: GAE bootstraps from V(s_T).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.nets import Actor

LOG_2PI = float(np.log(2 * np.pi))


class VCritic(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)

    def forward(self, x):
        return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(x))))).squeeze(-1)


class MAPPO:
    def __init__(self, n, obs_dim, act_dim=2, hidden=256, lr=5e-5, clip=0.2, epochs=45,
                 minibatch=4096, gamma=0.99, lam=0.9, ent_coef=0.0, vf_coef=1.0, max_grad=1.0,
                 init_log_std=0.0, share=True):
        self.n, self.obs_dim, self.act_dim = n, obs_dim, act_dim
        self.share = share
        k = 1 if share else n
        self.actors = [Actor(obs_dim, act_dim, hidden) for _ in range(k)]
        self.log_std = nn.Parameter(torch.full((k, act_dim), float(init_log_std)))
        self.critic = VCritic(n * obs_dim, hidden)
        params = [p for a in self.actors for p in a.parameters()] + [self.log_std]
        self.opt_pi = torch.optim.Adam(params, lr=lr)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.clip, self.epochs, self.minibatch = clip, epochs, minibatch
        self.gamma, self.lam, self.ent_coef, self.vf_coef = gamma, lam, ent_coef, vf_coef
        self.max_grad = max_grad

    def actor(self, i):
        return self.actors[0 if self.share else i]

    def mu(self, obs):
        """obs [B, n, obs_dim] -> pre-tanh mean [B, n, act_dim]."""
        outs = []
        for i in range(self.n):
            a = self.actor(i)
            x = F.relu(a.l2(F.relu(a.l1(obs[:, i]))))
            outs.append(a.l3(x))
        return torch.stack(outs, 1)

    def std(self):
        ls = self.log_std.clamp(-5, 2)
        return (ls if not self.share else ls.expand(self.n, -1)).exp()

    def logp(self, mu, u):
        std = self.std()
        return (-0.5 * ((u - mu) / std) ** 2 - std.log() - 0.5 * LOG_2PI).sum(-1)  # [B, n]

    @torch.no_grad()
    def act(self, obs):
        mu = self.mu(obs)
        u = mu + self.std() * torch.randn_like(mu)
        return u, torch.tanh(u), self.logp(mu, u)

    def update(self, obs, u, logp_old, rew, last_obs):
        """obs [T, E, n, d], u [T, E, n, a], logp_old [T, E, n], rew [T, E], last_obs [E, n, d]."""
        T, E = rew.shape
        with torch.no_grad():
            v = self.critic(obs.reshape(T, E, -1))
            v_last = self.critic(last_obs.reshape(E, -1))
            adv = torch.zeros_like(rew)
            gae = torch.zeros(E)
            for t in reversed(range(T)):
                nv = v_last if t == T - 1 else v[t + 1]
                delta = rew[t] + self.gamma * nv - v[t]
                gae = delta + self.gamma * self.lam * gae
                adv[t] = gae
            ret = adv + v
        N = T * E
        obs, u, logp_old = obs.reshape(N, self.n, -1), u.reshape(N, self.n, -1), logp_old.reshape(N, self.n)
        adv, ret = adv.reshape(N), ret.reshape(N)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = []
        for _ in range(self.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, self.minibatch):
                idx = perm[s:s + self.minibatch]
                ratio = (self.logp(self.mu(obs[idx]), u[idx]) - logp_old[idx]).exp()
                A = adv_n[idx].unsqueeze(1)
                pi_loss = -torch.min(ratio * A, ratio.clamp(1 - self.clip, 1 + self.clip) * A).mean()
                ent = (self.std().log() + 0.5 * (1 + LOG_2PI)).sum(-1).mean()
                self.opt_pi.zero_grad()
                (pi_loss - self.ent_coef * ent).backward()
                nn.utils.clip_grad_norm_([p for a in self.actors for p in a.parameters()] + [self.log_std],
                                         self.max_grad)
                self.opt_pi.step()
                v_loss = F.smooth_l1_loss(self.critic(obs[idx].reshape(len(idx), -1)), ret[idx])
                self.opt_v.zero_grad()
                (self.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
                self.opt_v.step()
                stats.append((pi_loss.item(), v_loss.item(), (ratio - 1).abs().mean().item()))
        return np.mean(stats, 0)
