"""Phase 1 Stage 2 evaluation: Phase 0 core random cells plus the Rphi cells, any team.

R cells use Phase 0's exact protocol (marl.attack.Attack, eval_curves.py seeds: pert_seed
[op, index of alpha in the 11-point grid, episode]), so Phase 0 teams reproduce their Phase 0
numbers. Rphi cells use the Stage 1 keys (eval_stage1.py), with the corruption applied to the
victims only. Scopes: one0, one1, one2 (worst victim = minimum over them, at analysis) and all.
Clean = alpha 0. Writes <out>/<label>.csv, one row per episode.
"""

import argparse
import csv
import json
import os

import numpy as np

from marl.attack import Attack
from marl.envs import make_env
from phase1.residual import corruption_sequence, load_team_any
from phase1.train_adversary_any import run_episode_team

T = 25
OPS = {"pr": 0, "nr": 1}
GRID = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def run_rphi(env, team, n, op, phi, alpha, victims, ep):
    rng = np.random.default_rng([2, int(phi * 100), int(alpha * 100), ep, 7])
    r = corruption_sequence("Rphi", phi, T, n, rng)
    coins = rng.random((T, n))
    vic = np.zeros(n, bool)
    vic[victims] = True
    obs = env.reset(seed=100_000 + ep)
    team.reset()
    total = 0.0
    for t in range(T):
        cmd = team.act(obs)
        if op == "nr":
            att = np.clip((1 - alpha) * cmd + alpha * r[t], -1, 1)
        else:
            att = np.where((coins[t] < alpha)[:, None], r[t], cmd)
        exe = np.where(vic[:, None], att, cmd).astype(np.float32)
        team.observe(cmd, exe)
        obs, _, team_r, _ = env.step(exe)
        total += team_r
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--label", required=True, help="<kind>_s<seed>")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    env = make_env(cfg.get("task", "spread"))
    n = env.n_team
    team = load_team_any(a.run_dir, n)
    scopes = [(f"one{j}", [j]) for j in range(n)] + [("all", list(range(n)))]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.label}.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "source", "op", "phi", "scope", "alpha", "episode", "return"])
        clean = Attack("nr", 0.0, [], "R", n)
        for ep in range(a.episodes):
            w.writerow([a.label, "clean", "none", 0, "none", 0, ep,
                        round(run_episode_team(env, team, clean, 100_000 + ep, [1, 0, ep]), 4)])
        for op in OPS:
            for scope, vic in scopes:
                for al in (0.1, 0.3, 0.5):
                    att = Attack(op, al, vic, "R", n)
                    for ep in range(a.episodes):
                        ret = run_episode_team(env, team, att, 100_000 + ep,
                                               [OPS[op], GRID.index(al), ep])
                        w.writerow([a.label, "R", op, 0, scope, al, ep, round(ret, 4)])
                    for phi in (0.9, 0.99):
                        for ep in range(a.episodes):
                            ret = run_rphi(env, team, n, op, phi, al, vic, ep)
                            w.writerow([a.label, "Rphi", op, phi, scope, al, ep, round(ret, 4)])
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
