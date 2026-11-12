"""Degradation curves under random (R) and gradient worst-case (G) attacks.

For each op in {pr, nr}, scope in {one agent j for each j, all agents} and
alpha in the grid, run episodes with common random numbers. Writes
<run_dir>/eval_curves.csv (one row per episode).
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from eval_critic import load_eval_critic
from marl.attack import Attack, load_team, run_episode
from marl.envs import make_env

OPS = {"pr": 0, "nr": 1}
SRC = {"R": 0, "G": 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--sources", default="R,G")
    p.add_argument("--alphas", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    p.add_argument("--episodes-R", type=int, default=100)
    p.add_argument("--episodes-G", type=int, default=50)
    a = p.parse_args()
    torch.set_num_threads(1)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    env = make_env(cfg["task"], prey_path=cfg.get("prey", "assets/tag_prey.pt"))
    n = env.n_team
    team = load_team(a.run_dir, n)
    alphas = [float(x) for x in a.alphas.split(",")]
    scopes = [(f"one{j}", [j]) for j in range(n)] + [("all", list(range(n)))]

    out = os.path.join(a.run_dir, "eval_curves.csv")
    with open(out + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "seed", "source", "op", "scope", "alpha", "episode", "return"])
        for src in a.sources.split(","):
            q = load_eval_critic(a.run_dir) if src == "G" else None
            n_ep = a.episodes_R if src == "R" else a.episodes_G
            for op in OPS:
                for scope, victims in scopes:
                    for ai, al in enumerate(alphas):
                        att = Attack(op, al, victims, src, n, eval_critic=q)
                        for ep in range(n_ep):
                            ret = run_episode(env, team, att, seed=100_000 + ep,
                                              pert_seed=[OPS[op], ai, ep])
                            w.writerow([cfg["task"], cfg["method"], cfg["seed"], src, op, scope,
                                        al, ep, round(ret, 4)])
    os.replace(out + ".tmp", out)


if __name__ == "__main__":
    main()
