"""MADDPG with method hooks (continuous actions in [-1,1]^2 per agent).

method  | what changes vs MADDPG
--------+-----------------------------------------------------------------
maddpg  | -
m3ddpg  | MAAL (Li et al. 2019): in agent i's critic target and actor loss,
        | other agents' actions get one l2-normalised gradient step that
        | lowers Q_i (step m3_eps), as in the original TF code.
ernie   | ERNIE-A, continuous adaptation of Bukharin et al. 2023 Sec 5.3:
        | critic loss += lam * max_{||d||_2<=eps} (Q_i(s,a) - Q_i(s,a+d))^2
        | over the joint action, solved by projected gradient ascent.
pr / nr | Tessler et al. 2019 PR-/NR-MDP applied per agent: each agent i has
        | an adversary nu_i. PR: with prob. alpha nu_i's action is executed;
        | NR: executed = (1-alpha) mu_i + alpha nu_i. Critics learn the joint
        | value; actors maximise, adversaries minimise (every adv_ratio actor
        | updates).
marta   | MARTA-AC adaptation (Mguni et al. 2025): a Switcher (DQN over
        | {none, agent 1..N}, reward -r - c*1[fault]) picks which agent
        | malfunctions each step; that agent executes a learned worst-case
        | fault policy sigma_i (continuous replacement for softmax(-Q*)).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from marl.nets import Actor, Critic, DiscreteQ, soft_update

METHODS = ["maddpg", "m3ddpg", "ernie", "pr", "nr", "marta"]


class Replay:
    def __init__(self, size, obs_dims, n, act_dim):
        self.size, self.ptr, self.full = size, 0, False
        self.obs = [np.zeros((size, d), np.float32) for d in obs_dims]
        self.obs2 = [np.zeros((size, d), np.float32) for d in obs_dims]
        self.act = np.zeros((size, n, act_dim), np.float32)
        self.rew = np.zeros((size, n), np.float32)
        self.team = np.zeros((size,), np.float32)
        self.k = np.zeros((size,), np.int64)

    def add(self, obs, act, rew, team, obs2, k=0):
        i = self.ptr
        for j in range(len(obs)):
            self.obs[j][i] = obs[j]
            self.obs2[j][i] = obs2[j]
        self.act[i], self.rew[i], self.team[i], self.k[i] = act, rew, team, k
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, b):
        idx = np.random.randint(0, len(self), b)
        t = torch.from_numpy
        return ([t(o[idx]) for o in self.obs], t(self.act[idx]), t(self.rew[idx]),
                t(self.team[idx]), [t(o[idx]) for o in self.obs2], t(self.k[idx]))


def maal(critic, obs_cat, act, i, eps):
    """M3DDPG: perturb agents j != i one normalised gradient step towards lower Q."""
    a = act.detach().clone().requires_grad_(True)
    q = critic(obs_cat, a.flatten(1)).mean()
    g = torch.autograd.grad(-q, a)[0]
    g = F.normalize(g, dim=-1)
    mask = torch.ones(act.shape[1], 1)
    mask[i] = 0.0
    return (act.detach() + eps * g * mask).clamp(-1, 1)


def ernie_reg(critic, obs_cat, act, eps, steps, step_size):
    """max_{||d||<=eps} (Q(s,a) - Q(s,a+d))^2 via PGA on the flattened joint action."""
    a0 = act.detach().flatten(1)
    with torch.no_grad():
        q0 = critic(obs_cat, a0)
    d = torch.randn_like(a0) * 1e-3
    for _ in range(steps):
        d.requires_grad_(True)
        gap = (critic(obs_cat, (a0 + d).clamp(-1, 1)) - q0).pow(2).sum()
        g = torch.autograd.grad(gap, d)[0]
        d = d.detach() + step_size * F.normalize(g, dim=-1)
        norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        d = d * torch.clamp(eps / norm, max=1.0)
    return (critic(obs_cat, a0) - critic(obs_cat, (a0 + d.detach()).clamp(-1, 1))).pow(2).mean()


class MADDPG:
    def __init__(self, obs_dims, method, cfg, rng):
        assert method in METHODS
        self.method, self.cfg, self.rng = method, cfg, rng
        self.n = len(obs_dims)
        self.obs_dims = obs_dims
        A = 2
        in_dim = sum(obs_dims) + self.n * A
        h = cfg["hidden"]
        self.actors = [Actor(d, A, h) for d in obs_dims]
        self.critics = [Critic(in_dim, h) for _ in obs_dims]
        self.actors_t = [copy.deepcopy(a) for a in self.actors]
        self.critics_t = [copy.deepcopy(c) for c in self.critics]
        self.a_opt = [torch.optim.Adam(a.parameters(), lr=cfg["lr_actor"]) for a in self.actors]
        self.c_opt = [torch.optim.Adam(c.parameters(), lr=cfg["lr_critic"]) for c in self.critics]
        self.adv = None
        if method in ("pr", "nr", "marta"):
            self.adv = [Actor(d, A, h) for d in obs_dims]
            self.adv_t = [copy.deepcopy(a) for a in self.adv]
            self.adv_opt = [torch.optim.Adam(a.parameters(), lr=cfg["lr_actor"]) for a in self.adv]
        if method == "marta":
            self.sw = DiscreteQ(sum(obs_dims), self.n + 1, h)
            self.sw_t = copy.deepcopy(self.sw)
            self.sw_opt = torch.optim.Adam(self.sw.parameters(), lr=cfg["lr_critic"])
        self.actor_updates = 0
        self.stats = {}

    # ------------------------------------------------------------------ acting
    def _noisy(self, nets, obs, noise):
        out = []
        with torch.no_grad():
            for net, o in zip(nets, obs):
                a = net(torch.as_tensor(o).unsqueeze(0))[0].numpy()
                if noise > 0:
                    a = a + self.rng.normal(0, noise, a.shape)
                out.append(np.clip(a, -1, 1))
        return np.stack(out).astype(np.float32)

    def act(self, obs, explore=True):
        """Executed joint action during training, plus switcher choice k (0 = none)."""
        noise = self.cfg["noise"] if explore else 0.0
        mu = self._noisy(self.actors, obs, noise)
        k = 0
        m, al = self.method, self.cfg["alpha"]
        if m in ("pr", "nr"):
            nu = self._noisy(self.adv, obs, noise)
            if m == "pr":
                swap = self.rng.random(self.n) < al
                mu = np.where(swap[:, None], nu, mu)
            else:
                mu = (1 - al) * mu + al * nu
        elif m == "marta":
            if explore and self.rng.random() < self.cfg["sw_eps"]:
                k = int(self.rng.integers(self.n + 1))
            else:
                with torch.no_grad():
                    k = int(self.sw(torch.as_tensor(np.concatenate(obs)).unsqueeze(0)).argmax())
            if k > 0:
                mu[k - 1] = self._noisy([self.adv[k - 1]], [obs[k - 1]], noise)[0]
        return np.clip(mu, -1, 1).astype(np.float32), k

    def team_policy(self, obs):
        """Nominal (unattacked, noise-free) joint action."""
        return self._noisy(self.actors, obs, 0.0)

    # ---------------------------------------------------------------- learning
    def _target_actions(self, obs2):
        with torch.no_grad():
            mu = torch.stack([a(o) for a, o in zip(self.actors_t, obs2)], 1)
            m, al = self.method, self.cfg["alpha"]
            if m in ("pr", "nr"):
                nu = torch.stack([a(o) for a, o in zip(self.adv_t, obs2)], 1)
                if m == "pr":
                    swap = (torch.rand(mu.shape[0], self.n, 1) < al).float()
                    mu = swap * nu + (1 - swap) * mu
                else:
                    mu = (1 - al) * mu + al * nu
            elif m == "marta":
                k = self.sw_t(torch.cat(obs2, -1)).argmax(-1)
                sig = torch.stack([a(o) for a, o in zip(self.adv_t, obs2)], 1)
                hit = F.one_hot(k, self.n + 1)[:, 1:].float().unsqueeze(-1)
                mu = hit * sig + (1 - hit) * mu
        return mu

    def update(self, buf):
        cfg, m, al = self.cfg, self.method, self.cfg["alpha"]
        obs, act, rew, team, obs2, k = buf.sample(cfg["batch"])
        oc, oc2 = torch.cat(obs, -1), torch.cat(obs2, -1)
        r_all = team.unsqueeze(1).expand(-1, self.n) if cfg["shared_reward"] else rew
        a2 = self._target_actions(obs2)
        do_actor = True
        for i in range(self.n):
            # ---- critic
            a2_i = maal(self.critics_t[i], oc2, a2, i, cfg["m3_eps"]) if m == "m3ddpg" else a2
            with torch.no_grad():
                y = r_all[:, i:i + 1] + cfg["gamma"] * self.critics_t[i](oc2, a2_i.flatten(1))
            q = self.critics[i](oc, act.flatten(1))
            c_loss = F.mse_loss(q, y)
            if m == "ernie":
                c_loss = c_loss + cfg["ernie_lam"] * ernie_reg(
                    self.critics[i], oc, act, cfg["ernie_eps"], cfg["ernie_steps"],
                    cfg["ernie_eps"] / 2)
            self.c_opt[i].zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.c_opt[i].step()

            # ---- actor
            others = act.clone()
            if m == "m3ddpg" and cfg.get("m3_actor", True):
                # m3_actor=False matches the released TF code, whose actor loss is built
                # from the unperturbed Q (the perturbed adv_q is computed but never used).
                others = maal(self.critics[i], oc, others, i, cfg["m3_eps"])
            own = self.actors[i](obs[i])
            if m == "nr":
                with torch.no_grad():
                    nu_i = self.adv[i](obs[i])
                own = (1 - al) * own + al * nu_i
            joint = torch.cat([others[:, :i], own.unsqueeze(1), others[:, i + 1:]], 1)
            a_loss = -self.critics[i](oc, joint.flatten(1)).mean()
            if m == "pr":
                a_loss = (1 - al) * a_loss
            self.a_opt[i].zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.a_opt[i].step()

            # ---- adversary / fault policy (minimises Q_i)
            adv_now = m == "marta" or (m in ("pr", "nr") and
                                       (self.actor_updates + 1) % cfg["adv_ratio"] == 0)
            if adv_now:
                nu = self.adv[i](obs[i])
                if m == "nr":
                    with torch.no_grad():
                        mu_i = self.actors[i](obs[i])
                    nu = (1 - al) * mu_i + al * nu
                joint = torch.cat([act[:, :i], nu.unsqueeze(1), act[:, i + 1:]], 1)
                v_loss = self.critics[i](oc, joint.flatten(1)).mean()
                if m == "pr":
                    v_loss = al * v_loss
                self.adv_opt[i].zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adv[i].parameters(), 0.5)
                self.adv_opt[i].step()
            self.stats.update({f"q{i}": q.mean().item(), f"closs{i}": c_loss.item()})
        self.actor_updates += 1

        if m == "marta":
            rs = -team - cfg["marta_cost"] * (k > 0).float()
            with torch.no_grad():
                ys = rs + cfg["gamma"] * self.sw_t(oc2).max(-1).values
            qs = self.sw(oc).gather(1, k.unsqueeze(1)).squeeze(1)
            s_loss = F.mse_loss(qs, ys)
            self.sw_opt.zero_grad()
            s_loss.backward()
            self.sw_opt.step()
            soft_update(self.sw_t, self.sw, cfg["tau"])

        for i in range(self.n):
            soft_update(self.actors_t[i], self.actors[i], cfg["tau"])
            soft_update(self.critics_t[i], self.critics[i], cfg["tau"])
            if self.adv is not None:
                soft_update(self.adv_t[i], self.adv[i], cfg["tau"])
