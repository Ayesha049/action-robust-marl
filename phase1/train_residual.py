"""Phase 1 Stage 1: train a residual (or the FT control) on a frozen Phase 0 team under NR.

Each episode draws one condition: clean with probability --p-clean, otherwise NR with a source
in {R, Rphi 0.9, Rphi 0.99} and alpha in {0.1, 0.3, 0.5}, uniformly; all agents are victims.
Episodes are 25 steps (MPE max_cycles), so replay stores whole episodes and the GRU is unrolled
from a zero state over each sampled episode (no burn-in needed).

Critic: one centralised Q(s_t, executed joint action, c_t) on the shared team reward, with
privileged context c_t = [alpha, r_t] (r_t = current replacement actions, zero when clean).
Target: y_t = R_t + gamma Q'(s_{t+1}, clip((1-alpha) mu'_{t+1} + alpha r_{t+1}), c_{t+1}),
bootstrapping at the time limit as Phase 0 does.
Actor (agent i): maximise Q with agent i's executed action recomputed from its current command
mu_i, other agents' executed actions from replay; plus c_r * ||mu_i - mu_b,i||^2 on clean
episodes. Base weights are frozen (residual) or copied and trained (FT).

Writes <out>/stage1.pt, config.json, train_log.csv, DONE.
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from marl.envs import make_env
from marl.nets import Critic, soft_update
from phase1.residual import corruption_sequence, load_base, make_actor

T = 25
SOURCES = [("R", 0.0), ("Rphi", 0.9), ("Rphi", 0.99)]
ALPHAS = [0.1, 0.3, 0.5]


class EpisodeReplay:
    def __init__(self, size, n, obs_dim):
        self.size, self.ptr, self.full = size, 0, False
        self.obs = np.zeros((size, T + 1, n, obs_dim), np.float32)
        self.cmd = np.zeros((size, T, n, 2), np.float32)
        self.exe = np.zeros((size, T, n, 2), np.float32)
        self.r = np.zeros((size, T + 1, n, 2), np.float32)
        self.rew = np.zeros((size, T), np.float32)
        self.alpha = np.zeros((size,), np.float32)
        self.clean = np.zeros((size,), np.float32)

    def add(self, obs, cmd, exe, r, rew, alpha, clean):
        i = self.ptr
        self.obs[i], self.cmd[i], self.exe[i], self.r[i], self.rew[i] = obs, cmd, exe, r, rew
        self.alpha[i], self.clean[i] = alpha, clean
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, b):
        idx = np.random.randint(0, len(self), b)
        f = lambda x: torch.from_numpy(x[idx])
        return (f(self.obs), f(self.cmd), f(self.exe), f(self.r), f(self.rew), f(self.alpha),
                f(self.clean))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, help="Phase 0 run dir, e.g. runs/spread__maddpg__s0")
    p.add_argument("--kind", choices=["residual", "ft"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20_000)  # 500k env steps
    p.add_argument("--p-clean", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=64, help="episodes per update")
    p.add_argument("--buffer", type=int, default=20_000, help="episodes")
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--critic-warmup", type=int, default=2_000, help="episodes of critic-only updates")
    p.add_argument("--beta-warmup", type=int, default=8_000, help="episodes to ramp beta 0 -> beta")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--c-r", type=float, default=1.0, help="clean-episode residual penalty")
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
    cin = n * od + n * 2 + 1 + n * 2
    critic = Critic(cin, a.critic_hidden)
    critic_t = copy.deepcopy(critic)
    c_opt = torch.optim.Adam(critic.parameters(), lr=a.lr_critic)
    buf = EpisodeReplay(a.buffer, n, od)

    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), task=base_cfg["task"], base_method=base_cfg["method"],
                   base_seed=base_cfg["seed"], git_sha=git_sha(), torch=torch.__version__),
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    logf = open(os.path.join(a.out, "train_log.csv"), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["episode", "beta", "ret_clean", "ret_attacked", "res_norm_clean",
                  "res_norm_attacked", "gate", "q", "closs", "sps"])
    hist = {"rc": [], "ra": [], "nc": [], "na": [], "g": []}
    stats = {"q": 0.0, "closs": 0.0}
    start = time.time()

    def ctx(alpha, r):  # (B, T', 1 + 2n)
        B, Tp = r.shape[:2]
        return torch.cat([alpha.view(B, 1, 1).expand(B, Tp, 1), r.reshape(B, Tp, -1)], -1)

    for ep in range(a.episodes):
        beta = a.beta * min(1.0, max(0.0, (ep - a.critic_warmup) / max(1, a.beta_warmup)))
        clean = rng.random() < a.p_clean
        if clean:
            alpha, r = 0.0, np.zeros((T + 1, n, 2), np.float32)
        else:
            src, phi = SOURCES[rng.integers(len(SOURCES))]
            alpha = ALPHAS[rng.integers(len(ALPHAS))]
            r = corruption_sequence(src, phi, T + 1, n, rng)
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        O = np.zeros((T + 1, n, od), np.float32)
        C = np.zeros((T, n, 2), np.float32)
        E = np.zeros((T, n, 2), np.float32)
        Rw = np.zeros((T,), np.float32)
        h = [torch.zeros(1, a.hidden) for _ in range(n)]
        pc, pe = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
        ret, rn = 0.0, []
        for t in range(T):
            O[t] = np.stack(obs)
            cmd = np.zeros((n, 2), np.float32)
            with torch.no_grad():
                for i in range(n):
                    tt = lambda x: torch.as_tensor(x).unsqueeze(0)
                    mu, mu_b, g, h[i] = actors[i].step(tt(O[t, i]), tt(pc[i]), tt(pe[i]), h[i], beta)
                    rn.append(float((mu - mu_b).norm()))
                    if g is not None:
                        hist["g"].append(float(g))
                    cmd[i] = mu[0].numpy()
            cmd = np.clip(cmd + rng.normal(0, a.noise, cmd.shape), -1, 1).astype(np.float32)
            exe = cmd if clean else np.clip((1 - alpha) * cmd + alpha * r[t], -1, 1)
            obs, _, team_r, done = env.step(exe)
            C[t], E[t], Rw[t] = cmd, exe, team_r
            pc, pe = cmd, exe.astype(np.float32)
            ret += team_r
        O[T] = np.stack(obs)
        buf.add(O, C, E, r, Rw, alpha, float(clean))
        hist["rc" if clean else "ra"].append(ret)
        hist["nc" if clean else "na"].append(float(np.mean(rn)))

        if len(buf) >= a.batch:
            o, c, e, rr, rew, al, cl = buf.sample(a.batch)
            B = o.shape[0]
            zero = torch.zeros(B, 1, n, 2)
            prev_c = torch.cat([zero, c], 1)  # (B, T+1, n, 2): evidence at step t = step t-1
            prev_e = torch.cat([zero, e], 1)
            oc = o.reshape(B, T + 1, -1)
            cx = ctx(al, rr)
            # ---- critic
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
            stats = {"q": q.mean().item(), "closs": closs.item()}
            # ---- actors
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
            log.writerow([ep + 1, round(beta, 3), m("rc"), m("ra"), m("nc"), m("na"), m("g"),
                          round(stats["q"], 3), round(stats["closs"], 4),
                          round((ep + 1) * T / (time.time() - start), 1)])
            logf.flush()
            for k in hist:
                hist[k] = hist[k][-500:]

    torch.save({"kind": a.kind, "hidden": a.hidden, "base_dir": os.path.abspath(a.base_dir),
                "actors": [x.state_dict() for x in actors]}, os.path.join(a.out, "stage1.pt"))
    open(os.path.join(a.out, "DONE"), "w").write(f"{time.time() - start:.0f} seconds\n")
    logf.close()


if __name__ == "__main__":
    main()
