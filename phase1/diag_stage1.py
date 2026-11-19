"""Phase 1 Stage 1 diagnostic: what does the residual change under attack?

Per team (residual / ft / maddpg / nr, one seed) and cell, logs every step and summarises:
  command      |cmd|, fraction of components saturated (|cmd| > 0.95), same for executed action
  residual     delta = cmd - mu_b (mu_b = frozen base command, recomputed for every team), |delta|
               by time step (detection latency), and a 2-coefficient least-squares fit
                   delta ~ k_amp * mu_b + k_comp * (-r_{t-1})          (t >= 1, NR cells)
               The NR oracle (Stage 0, r_hat = r_{t-1}) has k_amp = k_comp = alpha / (1 - alpha)
               before clipping: k_comp > 0 means within-step compensation, k_amp > 0 with
               k_comp ~ 0 means amplified commands without compensation.
  behaviour    speed, coverage (sum over landmarks of nearest-agent distance), agent-agent
               collisions (distance < 0.3), return
Ablation (residual only): "blind" feeds prev_exe := prev_cmd, removing the executed-action
evidence channel. Own velocity in the observation still reveals the executed action, so this
removes the direct channel only.

Cells: clean; NR R, Rphi 0.99 at alpha 0.3; PR-R 0.3; L-NR all alpha 0.3 with the team's own
adversaries (seeds 0, 1) and with the Phase 0 MADDPG adversaries of the same seed (one fixed
attack for every team). Random draws reuse eval_stage1 keys, so cells match Stage 1.
Writes <out>/diag_<label>.csv (per step) and prints nothing.
"""

import argparse
import csv
import os

import numpy as np
import torch

from marl.envs import make_env
from marl.nets import Actor
from phase1.residual import base_pre, corruption_sequence, load_base, load_team_any

T = 25
SRC_ID = {"clean": 0, "R": 1, "Rphi": 2}


def run(env, team, bases, n, op, source, phi, alpha, ep, adv=None, blind=False):
    if adv is None:
        rng = np.random.default_rng([SRC_ID[source], int(phi * 100), int(alpha * 100), ep, 7])
        r = corruption_sequence("R" if source == "clean" else source, phi, T, n, rng)
        coins = rng.random((T, n))
    obs = env.reset(seed=100_000 + ep)
    team.reset()
    rows, prev_r = [], np.zeros((n, 2), np.float32)
    for t in range(T):
        cmd = team.act(obs)
        with torch.no_grad():
            mub = np.stack([torch.tanh(base_pre(b, torch.as_tensor(obs[i]))).numpy()
                            for i, b in enumerate(bases)])
        if adv is not None:
            with torch.no_grad():
                rt = adv(torch.as_tensor(np.concatenate(obs)).unsqueeze(0))[0].numpy().reshape(n, 2)
        else:
            rt = r[t]
        if source == "clean":
            exe = cmd
        elif op == "nr":
            exe = np.clip((1 - alpha) * cmd + alpha * rt, -1, 1)
        else:
            exe = np.where((coins[t] < alpha)[:, None], rt, cmd)
        exe = exe.astype(np.float32)
        team.observe(cmd, cmd.copy() if blind else exe)
        pos = np.stack([o[2:4] for o in obs])
        lm = pos[0] + obs[0][4:10].reshape(3, 2)
        cover = np.linalg.norm(lm[:, None] - pos[None], axis=-1).min(1).sum()
        dd = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        coll = int(((dd < 0.3) & ~np.eye(n, dtype=bool)).sum() // 2)
        obs2, _, team_r, _ = env.step(exe)
        for i in range(n):
            rows.append([ep, t, i, *cmd[i], *mub[i], *exe[i], *rt[i], *prev_r[i],
                         float(np.linalg.norm(obs[i][0:2])), cover, coll, team_r])
        prev_r = rt.astype(np.float32)
        obs = obs2
    return rows


def load_adv(path, obs_dim, act_dim):
    a = Actor(obs_dim, act_dim, 128)
    a.load_state_dict(torch.load(path, map_location="cpu"))
    a.eval()
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--blind", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    env = make_env("spread")
    n = env.n_team
    team = load_team_any(a.run_dir, n)
    mdir = os.path.join(a.runs, f"spread__maddpg__s{a.seed}")
    bases = [load_base(os.path.join(mdir, f"actor_{i}.pt")) for i in range(n)]
    od = sum(env.obs_dims[:n])
    cells = [("clean", "none", "clean", 0.0, 0.0, None), ("R", "nr", "R", 0.0, 0.3, None),
             ("Rphi0.99", "nr", "Rphi", 0.99, 0.3, None), ("PR-R", "pr", "R", 0.0, 0.3, None)]
    for s in (0, 1):
        for tag, d in (("own", a.run_dir), ("maddpg", mdir)):
            f = os.path.join(d, "adv", f"nr_a0.3_all_s{s}.pt")
            cells.append((f"L-{tag}-s{s}", "nr", "L", 0.0, 0.3, load_adv(f, od, 2 * n)))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"diag_{a.label}{'_blind' if a.blind else ''}.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "blind", "cell", "alpha", "episode", "t", "agent", "cmd0", "cmd1",
                    "mub0", "mub1", "exe0", "exe1", "r0", "r1", "rprev0", "rprev1", "speed",
                    "cover", "coll", "team_r"])
        for cell, op, src, phi, al, adv in cells:
            for ep in range(a.episodes):
                for row in run(env, team, bases, n, op, src, phi, al, ep, adv, a.blind):
                    w.writerow([a.label, int(a.blind), cell, al] +
                               [round(x, 5) if isinstance(x, float) else x for x in row])
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
