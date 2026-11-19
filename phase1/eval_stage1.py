"""Phase 1 Stage 1 evaluation under random sources (R, Rphi), clean, and PR-R.

Works for Phase 0 run dirs (frozen MADDPG / NR-MADDPG teams) and Stage 1 run dirs (residual, FT).
Common random numbers: replacement sequences and PR coins depend only on (source, phi, alpha,
episode), with Phase 0 evaluation seeds (100_000 + episode), so every team faces identical draws.
Writes <out>/<label>.csv, one row per episode.
"""

import argparse
import csv
import json
import os

import numpy as np

from marl.envs import make_env
from phase1.residual import corruption_sequence, load_team_any

T = 25
SRC_ID = {"clean": 0, "R": 1, "Rphi": 2}


def run(env, team, n, op, source, phi, alpha, ep):
    rng = np.random.default_rng([SRC_ID[source], int(phi * 100), int(alpha * 100), ep, 7])
    r = corruption_sequence("R" if source == "clean" else source, phi, T, n, rng)
    coins = rng.random((T, n))
    obs = env.reset(seed=100_000 + ep)
    team.reset()
    total = 0.0
    for t in range(T):
        cmd = team.act(obs)
        if source == "clean":
            exe = cmd
        elif op == "nr":
            exe = np.clip((1 - alpha) * cmd + alpha * r[t], -1, 1)
        else:
            exe = np.where((coins[t] < alpha)[:, None], r[t], cmd)
        team.observe(cmd, exe.astype(np.float32))
        obs, _, team_r, done = env.step(exe)
        total += team_r
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--label", required=True, help="e.g. residual_s0, ft_s0, maddpg_s0")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    task = cfg.get("task", "spread")
    env = make_env(task)
    n = env.n_team
    team = load_team_any(a.run_dir, n)
    cells = [("none", "clean", 0.0, 0.0)]
    for al in (0.1, 0.3, 0.5):
        cells += [("nr", "R", 0.0, al), ("nr", "Rphi", 0.9, al), ("nr", "Rphi", 0.99, al)]
    cells += [("pr", "R", 0.0, 0.3)]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.label}.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "op", "source", "phi", "alpha", "episode", "return"])
        for op, src, phi, al in cells:
            for ep in range(a.episodes):
                w.writerow([a.label, op, src, phi, al, ep,
                            round(run(env, team, n, op, src, phi, al, ep), 4)])
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
