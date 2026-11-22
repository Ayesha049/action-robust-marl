"""Clean evaluation of a trained team, with Transport delivery stats.

Per episode (Phase 0 evaluation seeds 100_000 + episode): team return, and for transport whether
the package is on the goal at the end and the first step it got there (-1 if never).
Reference policies: --policy random (uniform actions), zero, or scripted (transport only: walk to
the point behind the package on the package-goal line, then push toward the goal), in place of
--run-dir. The scripted pusher is the reference for the Phase 2 Transport gate.
Writes <out>/<label>.csv and prints a one-line summary.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from marl.attack import load_team
from marl.envs import make_env


def scripted_pusher(obs):
    """VMAS transport obs: pos 0:2, vel 2:4, package - goal 4:6, package - agent 6:8, ..."""
    out = []
    for o in obs:
        pg, pa = o[4:6], o[6:8]
        u = -pg / (np.linalg.norm(pg) + 1e-8)          # unit vector package -> goal
        to = pa - 0.11 * u                              # agent -> point behind the package
        d = np.linalg.norm(to)
        a = to / d if d > 0.05 else u + 5 * to
        out.append(np.clip(a, -1, 1))
    return np.stack(out).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="")
    p.add_argument("--task", default="")
    p.add_argument("--policy", default="", choices=["", "random", "zero", "scripted"])
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(1)
    task = json.load(open(os.path.join(a.run_dir, "config.json")))["task"] if a.run_dir else a.task
    env = make_env(task)
    n = env.n_team
    rng = np.random.default_rng(0)
    if a.run_dir:
        team = load_team(a.run_dir, n)
    elif a.policy == "random":
        team = lambda obs: rng.uniform(-1, 1, (n, 2)).astype(np.float32)  # noqa: E731
    elif a.policy == "scripted":
        team = scripted_pusher
    else:
        team = lambda obs: np.zeros((n, 2), np.float32)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for e in range(a.episodes):
        obs = env.reset(seed=100_000 + e)
        total, done, t, first = 0.0, False, 0, -1
        while not done:
            obs, _, r, done = env.step(team(obs))
            total += r
            if task.startswith("transport") and first < 0 and obs[0][10] > 0.5:
                first = t
            t += 1
        on_goal = int(obs[0][10] > 0.5) if task.startswith("transport") else -1
        rows.append([a.label, e, round(total, 4), on_goal, first])
    with open(os.path.join(a.out, f"{a.label}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "episode", "return", "on_goal_end", "first_delivery"])
        w.writerows(rows)
    R = np.array([r[2] for r in rows])
    D = np.array([r[3] for r in rows])
    F = np.array([r[4] for r in rows])
    print(f"{a.label}: return {R.mean():.2f} (sd {R.std():.2f}); on goal at end {D.mean():.2f}; "
          f"ever delivered {(F >= 0).mean():.2f}")


if __name__ == "__main__":
    main()
