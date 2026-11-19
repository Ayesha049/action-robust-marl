"""Phase 1 Stage 2: train RC / RC-fast / FT / REC / RES-MLP on a frozen Phase 0 team under NR,
with a co-trained, rate-bounded learned adversary in the training mix.

Stage 1 (train_residual.py) plus:
  * Training mix per episode: clean (--p-clean), random NR (R, Rphi 0.9, Rphi 0.99), or learned
    NR, the last two splitting the remainder equally (framework doc section 4). alpha is drawn
    from {0.1, 0.3, 0.5}; all agents are victims.
  * Learned adversary (framework doc sections 4 and 7.3): DDPG on the negated team reward, input
    [global obs, r_{t-1}, alpha], output a target that is projected onto the rate bound
        r_t = clip(r_{t-1} + proj_eps(target - r_{t-1}), -1, 1),   ||r_t,i - r_{t-1},i|| <= eps
    per victim, with r_{-1} = 0 and eps = --rate (0.5, shared by every kind, as in GRAD).
    Updated in alternation with the team: after each learned episode, 25 critic steps (one per
    environment step, the Phase 0 adversary ratio) with an actor step every second one, batch
    256, once 10k transitions are stored. A snapshot is added to a pool every --pool-every
    episodes (latest --pool-size kept); a learned episode uses the current adversary (plus
    exploration noise) with probability 0.5, otherwise a pool member.
Everything else (critic with privileged context [alpha, r_t], whole-episode replay, clean
penalty against the frozen base, beta ramp, hyperparameters) is as in Stage 1, untuned.

Writes <out>/stage2.pt, adversary.pt, config.json, train_log.csv, DONE.
"""

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from marl.envs import make_env
from marl.nets import Actor, Critic, soft_update
from phase1.residual import KINDS, corruption_sequence, load_base, make_actor
from phase1.train_residual import ALPHAS, SOURCES, T, EpisodeReplay, git_sha


def project(r_prev, target, eps):
    """Per-agent rate bound; r_prev, target (..., n, 2)."""
    d = target - r_prev
    nrm = d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (r_prev + d * torch.clamp(eps / nrm, max=1.0)).clamp(-1, 1)


class Adversary:
    def __init__(self, n, od, eps, lr=1e-3, hidden=128, gamma=0.95, tau=0.005, size=300_000):
        self.n, self.eps, self.gamma, self.tau = n, eps, gamma, tau
        self.sd = n * od + 2 * n + 1
        self.actor = Actor(self.sd, 2 * n, hidden)
        self.critic = Critic(self.sd + 2 * n, hidden)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.S = np.zeros((size, self.sd), np.float32)
        self.S2 = np.zeros_like(self.S)
        self.U = np.zeros((size, 2 * n), np.float32)
        self.R = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.count = size, 0, 0
        self.pool = []

    def state(self, obs, r_prev, alpha):
        return np.concatenate([np.concatenate(obs), r_prev.reshape(-1), [alpha]]).astype(np.float32)

    def policy(self, net, s):
        s = torch.as_tensor(s)
        rp = s[..., self.sd - 1 - 2 * self.n:self.sd - 1].reshape(*s.shape[:-1], self.n, 2)
        tgt = net(s).reshape(*s.shape[:-1], self.n, 2)
        return project(rp, tgt, self.eps).reshape(*s.shape[:-1], 2 * self.n)

    def act(self, net, s, noise, rng):
        with torch.no_grad():
            s_t = torch.as_tensor(s).unsqueeze(0)
            rp = s_t[:, self.sd - 1 - 2 * self.n:self.sd - 1].reshape(1, self.n, 2)
            tgt = net(s_t).reshape(1, self.n, 2)
            if noise:
                tgt = tgt + torch.as_tensor(rng.normal(0, noise, tgt.shape), dtype=torch.float32)
            return project(rp, tgt, self.eps)[0].numpy().astype(np.float32)

    def add(self, s, u, r, s2):
        i = self.ptr
        self.S[i], self.U[i], self.R[i], self.S2[i] = s, u, r, s2
        self.ptr = (i + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def update(self, steps=25, batch=256):
        if self.count < 10_000:
            return None
        for k in range(steps):
            idx = np.random.randint(0, self.count, batch)
            s, u, r, s2 = (torch.as_tensor(x[idx]) for x in (self.S, self.U, self.R, self.S2))
            with torch.no_grad():
                y = r + self.gamma * self.critic_t(s2, self.policy(self.actor_t, s2))
            loss = F.mse_loss(self.critic(s, u), y)
            self.c_opt.zero_grad()
            loss.backward()
            self.c_opt.step()
            if k % 2 == 0:
                pl = -self.critic(s, self.policy(self.actor, s)).mean()
                self.a_opt.zero_grad()
                pl.backward()
                self.a_opt.step()
                soft_update(self.actor_t, self.actor, self.tau)
                soft_update(self.critic_t, self.critic, self.tau)
        return loss.item()

    def snapshot(self, k):
        self.pool.append(copy.deepcopy(self.actor).eval())
        self.pool = self.pool[-k:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, help="Phase 0 run dir, e.g. runs/spread__pr__s0")
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--p-clean", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=20_000)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--critic-warmup", type=int, default=2_000)
    p.add_argument("--beta-warmup", type=int, default=8_000)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--c-r", type=float, default=1.0)
    p.add_argument("--rate", type=float, default=0.5, help="adversary rate bound eps")
    p.add_argument("--adv-noise", type=float, default=0.1)
    p.add_argument("--pool-every", type=int, default=1_000)
    p.add_argument("--pool-size", type=int, default=5)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    base_cfg = json.load(open(os.path.join(a.base_dir, "config.json")))
    env = make_env(base_cfg["task"], prey_path=base_cfg.get("prey", "assets/tag_prey.pt"))
    n, od = env.n_team, env.obs_dims[0]
    assert all(d == od for d in env.obs_dims[:n])
    actors = [make_actor(a.kind, load_base(os.path.join(a.base_dir, f"actor_{i}.pt")), a.hidden)
              for i in range(n)]
    actors_t = [copy.deepcopy(x) for x in actors]
    a_opt = [torch.optim.Adam(x.trainable(), lr=a.lr_actor) for x in actors]
    critic = Critic(n * od + n * 2 + 1 + n * 2, a.critic_hidden)
    critic_t = copy.deepcopy(critic)
    c_opt = torch.optim.Adam(critic.parameters(), lr=a.lr_critic)
    buf = EpisodeReplay(a.buffer, n, od)
    adv = Adversary(n, od, a.rate)
    sdim = getattr(actors[0], "state_dim", getattr(actors[0], "hidden", 1))

    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), task=base_cfg["task"], base_method=base_cfg["method"],
                   base_seed=base_cfg["seed"], git_sha=git_sha(), torch=torch.__version__),
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    logf = open(os.path.join(a.out, "train_log.csv"), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["episode", "beta", "ret_clean", "ret_random", "ret_learned", "res_norm_clean",
                  "res_norm_attacked", "gate", "q", "closs", "adv_closs", "adv_rate", "sps"])
    hist = {k: [] for k in ("rc", "rr", "rl", "nc", "na", "g", "rate")}
    stats = {"q": 0.0, "closs": 0.0, "adv": 0.0}
    start = time.time()

    def ctx(alpha, r):
        B, Tp = r.shape[:2]
        return torch.cat([alpha.view(B, 1, 1).expand(B, Tp, 1), r.reshape(B, Tp, -1)], -1)

    p_rand = (1 - a.p_clean) / 2
    for ep in range(a.episodes):
        beta = a.beta * min(1.0, max(0.0, (ep - a.critic_warmup) / max(1, a.beta_warmup)))
        u = rng.random()
        cond = "clean" if u < a.p_clean else "random" if u < a.p_clean + p_rand else "learned"
        clean = cond == "clean"
        alpha = 0.0 if clean else ALPHAS[rng.integers(len(ALPHAS))]
        r = np.zeros((T + 1, n, 2), np.float32)
        if cond == "random":
            src, phi = SOURCES[rng.integers(len(SOURCES))]
            r = corruption_sequence(src, phi, T + 1, n, rng)
        if cond == "learned":
            cur = not adv.pool or rng.random() < 0.5
            net = adv.actor if cur else adv.pool[rng.integers(len(adv.pool))]
            anoise = a.adv_noise if cur else 0.0
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        O = np.zeros((T + 1, n, od), np.float32)
        C = np.zeros((T, n, 2), np.float32)
        E = np.zeros((T, n, 2), np.float32)
        Rw = np.zeros((T,), np.float32)
        h = [torch.zeros(1, sdim) for _ in range(n)]
        pc, pe = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
        r_prev = np.zeros((n, 2), np.float32)
        ret, rn, trans = 0.0, [], []
        for t in range(T):
            O[t] = np.stack(obs)
            if cond == "learned":
                s_adv = adv.state(obs, r_prev, alpha)
                r[t] = adv.act(net, s_adv, anoise, rng)
            cmd = np.zeros((n, 2), np.float32)
            with torch.no_grad():
                for i in range(n):
                    tt = lambda x: torch.as_tensor(x).unsqueeze(0)
                    mu, mu_b, g, h[i] = actors[i].step(tt(O[t, i]), tt(pc[i]), tt(pe[i]), h[i], beta)
                    rn.append(float((mu - mu_b).norm()))
                    if g is not None:
                        hist["g"].append(float(g.mean()))
                    cmd[i] = mu[0].numpy()
            cmd = np.clip(cmd + rng.normal(0, a.noise, cmd.shape), -1, 1).astype(np.float32)
            exe = cmd if clean else np.clip((1 - alpha) * cmd + alpha * r[t], -1, 1)
            obs, _, team_r, done = env.step(exe)
            C[t], E[t], Rw[t] = cmd, exe, team_r
            pc, pe = cmd, exe.astype(np.float32)
            ret += team_r
            if cond == "learned":
                hist["rate"].append(float(np.linalg.norm(r[t] - r_prev, axis=-1).mean()))
                r_prev = r[t].copy()
                trans.append((s_adv, r[t].reshape(-1), -team_r, adv.state(obs, r_prev, alpha)))
        O[T] = np.stack(obs)
        if cond == "learned":  # r_T for the critic target at the last step
            r[T] = adv.act(net, adv.state(obs, r_prev, alpha), 0.0, rng)
            for tr in trans:
                adv.add(*tr)
            out = adv.update()
            if out is not None:
                stats["adv"] = out
        if (ep + 1) % a.pool_every == 0:
            adv.snapshot(a.pool_size)
        buf.add(O, C, E, r, Rw, alpha, float(clean))
        hist[{"clean": "rc", "random": "rr", "learned": "rl"}[cond]].append(ret)
        hist["nc" if clean else "na"].append(float(np.mean(rn)))

        if len(buf) >= a.batch:
            o, c, e, rr, rew, al, cl = buf.sample(a.batch)
            B = o.shape[0]
            zero = torch.zeros(B, 1, n, 2)
            prev_c = torch.cat([zero, c], 1)
            prev_e = torch.cat([zero, e], 1)
            oc = o.reshape(B, T + 1, -1)
            cx = ctx(al, rr)
            with torch.no_grad():
                mu_t = torch.stack([actors_t[i].unroll(o[:, :, i], prev_c[:, :, i], prev_e[:, :, i],
                                                       beta)[0] for i in range(n)], 2)
                exe2 = ((1 - al).view(B, 1, 1, 1) * mu_t + al.view(B, 1, 1, 1) * rr).clamp(-1, 1)
                q2 = critic_t(oc[:, 1:], torch.cat([exe2[:, 1:].reshape(B, T, -1), cx[:, 1:]], -1))
                y = rew.unsqueeze(-1) + a.gamma * q2
            q = critic(oc[:, :T], torch.cat([e.reshape(B, T, -1), cx[:, :T]], -1))
            closs = F.mse_loss(q, y)
            c_opt.zero_grad()
            closs.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            c_opt.step()
            stats.update(q=q.mean().item(), closs=closs.item())
            if ep >= a.critic_warmup:
                for i in range(n):
                    mu, mu_b, _ = actors[i].unroll(o[:, :T, i], prev_c[:, :T, i], prev_e[:, :T, i],
                                                   beta)
                    own = ((1 - al).view(B, 1, 1) * mu + al.view(B, 1, 1) * rr[:, :T, i]).clamp(-1, 1)
                    joint = torch.cat([e[:, :, :i], own.unsqueeze(2), e[:, :, i + 1:]], 2)
                    qa = critic(oc[:, :T], torch.cat([joint.reshape(B, T, -1), cx[:, :T]], -1))
                    pen = (cl.view(B, 1) * (mu - mu_b).pow(2).sum(-1)).sum() / (cl.sum() * T + 1e-8)
                    loss = -qa.mean() + a.c_r * pen
                    a_opt[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actors[i].trainable(), 0.5)
                    a_opt[i].step()
                for i in range(n):
                    for tp, sp in zip(actors_t[i].trainable(), actors[i].trainable()):
                        tp.data.mul_(1 - a.tau).add_(a.tau * sp.data)
            soft_update(critic_t, critic, a.tau)

        if (ep + 1) % 500 == 0:
            m = lambda k: round(float(np.mean(hist[k][-500:])), 4) if hist[k] else ""
            log.writerow([ep + 1, round(beta, 3), m("rc"), m("rr"), m("rl"), m("nc"), m("na"),
                          m("g"), round(stats["q"], 3), round(stats["closs"], 4),
                          round(stats["adv"], 4), m("rate"),
                          round((ep + 1) * T / (time.time() - start), 1)])
            logf.flush()
            for k in hist:
                hist[k] = hist[k][-500:]

    torch.save({"kind": a.kind, "hidden": a.hidden, "base_dir": os.path.abspath(a.base_dir),
                "actors": [x.state_dict() for x in actors]}, os.path.join(a.out, "stage2.pt"))
    torch.save({"actor": adv.actor.state_dict(), "pool": [x.state_dict() for x in adv.pool],
                "rate": a.rate}, os.path.join(a.out, "adversary.pt"))
    open(os.path.join(a.out, "DONE"), "w").write(f"{time.time() - start:.0f} seconds\n")
    logf.close()


if __name__ == "__main__":
    main()
