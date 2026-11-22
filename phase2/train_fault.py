"""Phase 2 Stage 1: train RC / RC-blind / REC / FT on a frozen Phase 0 team under persistent
actuator faults (phase2/faults.py).

Structure as Phase 1 Stage 1 (phase1/train_residual.py), with the NR channel replaced by the
fault channel. Every fault kind is affine per step: exe_j = clip(M_t,j cmd_j + b_t,j), with
M = I, b = 0 before the onset and for non-victims, so one differentiable channel serves all kinds.

Training mix per episode (Stage 1 development default, 2026-11-22):
  clean with probability --p-clean; otherwise one fault cell, drawn as
    kind      rot 1/2; gain, bias, drift 1/6 each        (stuck is a held-out negative control)
    strength  rot theta ~ U[0, 180] deg (random sign); gain g ~ U[0.3, 1]; bias |b| ~ U[0, 0.9];
              drift eta ~ U[0, 0.04]
    victims   one random agent (1/2) or all agents (1/2)
    onset     t0 ~ U{0, ..., 12}, as in Stage 0
Critic: Q(s_t, executed joint action, c_t) with privileged context c_t = [M_t, b_t] for every
agent (zero offsets and identity when clean). Target uses clip(M_{t+1} mu'_{t+1} + b_{t+1}).
Actor (agent i): maximise Q with agent i's executed action recomputed through the fault channel
from its current command, the others' executed actions from replay; plus c_r ||mu_i - mu_b,i||^2
on clean episodes.

Kinds:
  residual  RC: frozen base + gated, bounded GRU residual over [o, a_{t-1}, a~_{t-1}, mu_b];
            head in action space by default (mu = clip(mu_b + g beta 2 tanh(D))), --space pre for
            the Phase 1 form (run 1 used pre: it cannot reach the corrections on a saturated base)
  blind     RC-blind: the same network, with a~_{t-1} replaced by a_{t-1} (no executed-action
            evidence), in training and evaluation
  rec       REC: RC's network with the base weights trained too
  ft        FT: the base MLP fine-tuned, no memory

Writes <out>/stage1.pt, config.json, train_log.csv, DONE. The checkpoint loads with
phase1.residual.load_team_any (kind "blind" is stored as "residual" plus blind=True).
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
from marl.nets import Critic, soft_update
from phase1.residual import load_base, make_actor
from phase1.train_residual import git_sha
from phase2.faults import ONSET_MAX, Fault, affine

T = 25
KINDS = ("residual", "blind", "rec", "ft")
MIX = (("rot", 0.5), ("gain", 1 / 6), ("bias", 1 / 6), ("drift", 1 / 6))
RANGES = {"rot": (0.0, 180.0), "gain": (0.3, 1.0), "bias": (0.0, 0.9), "drift": (0.0, 0.04)}


def sample_fault(rng, n):
    kind = MIX[rng.choice(len(MIX), p=[w for _, w in MIX])][0]
    lo, hi = RANGES[kind]
    victims = [int(rng.integers(n))] if rng.random() < 0.5 else list(range(n))
    return Fault(kind, rng.uniform(lo, hi), victims, n)


def apply(M, b, cmd):
    """M (..., 2, 2), b (..., 2), cmd (..., 2) -> clip(M cmd + b)."""
    return (torch.einsum("...ij,...j->...i", M, cmd) + b).clamp(-1, 1)


class EpisodeReplay:
    def __init__(self, size, n, od):
        self.size, self.ptr, self.full = size, 0, False
        self.obs = np.zeros((size, T + 1, n, od), np.float32)
        self.cmd = np.zeros((size, T, n, 2), np.float32)
        self.exe = np.zeros((size, T, n, 2), np.float32)
        self.M = np.zeros((size, T + 1, n, 2, 2), np.float32)
        self.b = np.zeros((size, T + 1, n, 2), np.float32)
        self.rew = np.zeros((size, T), np.float32)
        self.clean = np.zeros((size,), np.float32)

    def add(self, obs, cmd, exe, M, b, rew, clean):
        i = self.ptr
        self.obs[i], self.cmd[i], self.exe[i], self.M[i], self.b[i] = obs, cmd, exe, M, b
        self.rew[i], self.clean[i] = rew, clean
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, k):
        idx = np.random.randint(0, len(self), k)
        f = lambda x: torch.from_numpy(x[idx])  # noqa: E731
        return tuple(f(x) for x in (self.obs, self.cmd, self.exe, self.M, self.b, self.rew,
                                    self.clean))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, help="Phase 0 run dir, e.g. runs/tag__maddpg__s0")
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
    p.add_argument("--space", choices=["pre", "action"], default="action",
                   help="residual head: Phase 1 pre-tanh, or action space (run 2 on, 2026-11-22)")
    p.add_argument("--res-scale", type=float, default=2.0, help="action-space residual bound")
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    base_cfg = json.load(open(os.path.join(a.base_dir, "config.json")))
    assert float(base_cfg.get("act_cap", 1.0)) == 1.0, "capped bases are not supported here"
    env = make_env(base_cfg["task"], prey_path=base_cfg.get("prey", "assets/tag_prey.pt"))
    n, od = env.n_team, env.obs_dims[0]
    assert all(d == od for d in env.obs_dims[:n])
    blind = a.kind == "blind"
    net_kind = "residual" if blind else a.kind
    actors = [make_actor(net_kind, load_base(os.path.join(a.base_dir, f"actor_{i}.pt")), a.hidden,
                         a.space, a.res_scale) for i in range(n)]
    actors_t = [copy.deepcopy(x) for x in actors]
    a_opt = [torch.optim.Adam(x.trainable(), lr=a.lr_actor) for x in actors]
    critic = Critic(n * od + n * 2 + n * 6, a.critic_hidden)
    critic_t = copy.deepcopy(critic)
    c_opt = torch.optim.Adam(critic.parameters(), lr=a.lr_critic)
    buf = EpisodeReplay(a.buffer, n, od)
    sdim = getattr(actors[0], "state_dim", getattr(actors[0], "hidden", 1))

    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), task=base_cfg["task"], method=a.kind, base_method=base_cfg["method"],
                   base_seed=base_cfg["seed"], mix=MIX, ranges=RANGES, onset_max=ONSET_MAX,
                   git_sha=git_sha(), torch=torch.__version__),
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    logf = open(os.path.join(a.out, "train_log.csv"), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["episode", "beta", "ret_clean", "ret_fault", "ret_rot", "res_norm_clean",
                  "res_norm_fault", "gate", "q", "closs", "sps"])
    hist = {k: [] for k in ("rc", "rf", "rr", "nc", "nf", "g")}
    stats = {"q": 0.0, "closs": 0.0}
    start = time.time()

    def ctx(M, b):
        B, Tp = b.shape[:2]
        return torch.cat([M.reshape(B, Tp, -1), b.reshape(B, Tp, -1)], -1)

    for ep in range(a.episodes):
        beta = a.beta * min(1.0, max(0.0, (ep - a.critic_warmup) / max(1, a.beta_warmup)))
        clean = rng.random() < a.p_clean
        fault = None if clean else sample_fault(rng, n)
        fep = fault.sample(rng) if fault else None
        if fault is None:
            Ms = np.tile(np.eye(2, dtype=np.float32), (T + 1, n, 1, 1))
            bs = np.zeros((T + 1, n, 2), np.float32)
        else:
            Ms, bs = (np.stack(x) for x in zip(*(affine(fault, fep, t) for t in range(T + 1))))
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        O = np.zeros((T + 1, n, od), np.float32)
        C = np.zeros((T, n, 2), np.float32)
        E = np.zeros((T, n, 2), np.float32)
        Rw = np.zeros((T,), np.float32)
        h = [torch.zeros(1, sdim) for _ in range(n)]
        pc, pe = np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32)
        ret, rn = 0.0, []
        for t in range(T):
            O[t] = np.stack(obs)
            cmd = np.zeros((n, 2), np.float32)
            with torch.no_grad():
                for i in range(n):
                    tt = lambda x: torch.as_tensor(x).unsqueeze(0)  # noqa: E731
                    mu, mu_b, g, h[i] = actors[i].step(tt(O[t, i]), tt(pc[i]),
                                                       tt(pc[i] if blind else pe[i]), h[i], beta)
                    rn.append(float((mu - mu_b).norm()))
                    if g is not None:
                        hist["g"].append(float(g.mean()))
                    cmd[i] = mu[0].numpy()
            cmd = np.clip(cmd + rng.normal(0, a.noise, cmd.shape), -1, 1).astype(np.float32)
            exe = np.clip(np.einsum("nij,nj->ni", Ms[t], cmd) + bs[t], -1, 1).astype(np.float32)
            obs, _, team_r, done = env.step(exe)
            C[t], E[t], Rw[t] = cmd, exe, team_r
            pc, pe = cmd, exe
            ret += team_r
        O[T] = np.stack(obs)
        buf.add(O, C, E, Ms, bs, Rw, float(clean))
        hist["rc" if clean else "rf"].append(ret)
        if fault is not None and fault.kind == "rot":
            hist["rr"].append(ret)
        hist["nc" if clean else "nf"].append(float(np.mean(rn)))

        if len(buf) >= a.batch:
            o, c, e, M, bb, rew, cl = buf.sample(a.batch)
            B = o.shape[0]
            zero = torch.zeros(B, 1, n, 2)
            prev_c = torch.cat([zero, c], 1)
            prev_e = prev_c if blind else torch.cat([zero, e], 1)
            oc = o.reshape(B, T + 1, -1)
            cx = ctx(M, bb)
            with torch.no_grad():
                mu_t = torch.stack([actors_t[i].unroll(o[:, :, i], prev_c[:, :, i], prev_e[:, :, i],
                                                       beta)[0] for i in range(n)], 2)
                exe2 = apply(M, bb, mu_t)
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
                    own = apply(M[:, :T, i], bb[:, :T, i], mu)
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
            m = lambda k: round(float(np.mean(hist[k][-500:])), 4) if hist[k] else ""  # noqa: E731
            log.writerow([ep + 1, round(beta, 3), m("rc"), m("rf"), m("rr"), m("nc"), m("nf"),
                          m("g"), round(stats["q"], 3), round(stats["closs"], 4),
                          round((ep + 1) * T / (time.time() - start), 1)])
            logf.flush()
            for k in hist:
                hist[k] = hist[k][-500:]

    torch.save({"kind": net_kind, "blind": blind, "hidden": a.hidden, "space": a.space,
                "res_scale": a.res_scale,
                "base_dir": os.path.abspath(a.base_dir),
                "actors": [x.state_dict() for x in actors]}, os.path.join(a.out, "stage1.pt"))
    open(os.path.join(a.out, "DONE"), "w").write(f"{time.time() - start:.0f} seconds\n")
    logf.close()


if __name__ == "__main__":
    main()
