"""Phase 1: does cancelling carried-over momentum explain the residual's gain under i.i.d. noise?

MPE (mpe2 core): v <- 0.75 v + 0.5 u (damping 0.25, sensitivity 5, dt 0.1, mass 1), so the
execution error of step t-1, e = a~_{t-1} - a*_{t-1} (a* = base command), is still in the
velocity and carries 0.75 of it into the next step. Oracle command on the frozen base team:

    cmd_t = clip((a*_t - alpha * r_hat - c * 0.75 * e_{t-1}) / (1 - alpha))      (NR)
    cmd_t = clip(a*_t - c * 0.75 * e_{t-1})                                       (PR)

r_hat in {0 ("none"), r_{t-1} ("last")}; c in {0, 0.5, 1, 1.5}. (none, c=0) is the base with the
NR rescaling only, (last, c=0) is the Stage 0 "last" oracle. Only the executed action and the
team's own commands are used, so the oracle is causal; alpha is known (oracle).
Cells and random draws reuse eval_stage1 keys; L-NR uses the team's own Phase 0 adversaries.
Writes <out>/mom_<label>.csv (one row per episode).
"""

import argparse
import csv
import os

import numpy as np
import torch

from marl.attack import load_team
from marl.envs import make_env
from marl.nets import Actor
from phase1.residual import corruption_sequence

T = 25
SRC_ID = {"R": 1, "Rphi": 2}


def run(env, team, n, op, source, phi, alpha, ep, rhat, c, adv=None):
    if adv is None:
        rng = np.random.default_rng([SRC_ID[source], int(phi * 100), int(alpha * 100), ep, 7])
        r = corruption_sequence(source, phi, T, n, rng)
        coins = rng.random((T, n))
    obs = env.reset(seed=100_000 + ep)
    prev_r = np.zeros((n, 2), np.float32)
    e = np.zeros((n, 2), np.float32)
    total = 0.0
    for t in range(T):
        a_star = team(obs)
        m = c * 0.75 * e
        if op == "nr":
            rh = prev_r if rhat == "last" else 0.0
            cmd = np.clip((a_star - alpha * rh - m) / (1 - alpha), -1, 1)
        else:
            cmd = np.clip(a_star - m, -1, 1)
        if adv is not None:
            with torch.no_grad():
                rt = adv(torch.as_tensor(np.concatenate(obs)).unsqueeze(0))[0].numpy().reshape(n, 2)
        else:
            rt = r[t]
        if op == "nr":
            exe = np.clip((1 - alpha) * cmd + alpha * rt, -1, 1)
        else:
            exe = np.where((coins[t] < alpha)[:, None], rt, cmd)
        e = (exe - a_star).astype(np.float32)
        prev_r = rt.astype(np.float32)
        obs, _, team_r, _ = env.step(exe)
        total += team_r
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(1)
    env = make_env("spread")
    n = env.n_team
    team = load_team(a.run_dir, n)
    od = sum(env.obs_dims[:n])
    cells = [("nr", "R", 0.0, al, None) for al in (0.1, 0.3, 0.5)]
    cells += [("nr", "Rphi", ph, 0.3, None) for ph in (0.9, 0.99)]
    cells += [("pr", "R", 0.0, 0.3, None)]
    for s in (0, 1):
        adv = Actor(od, 2 * n, 128)
        adv.load_state_dict(torch.load(os.path.join(a.run_dir, "adv", f"nr_a0.3_all_s{s}.pt"),
                                       map_location="cpu"))
        adv.eval()
        cells.append(("nr", f"L{s}", 0.0, 0.3, adv))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"mom_{a.label}.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "op", "source", "phi", "alpha", "rhat", "c", "episode", "return"])
        for ep in range(a.episodes):  # clean reference
            w.writerow([a.label, "none", "clean", 0, 0, "none", 0, ep,
                        round(run(env, team, n, "nr", "R", 0.0, 0.0, ep, "none", 0.0), 4)])
        for op, src, phi, al, adv in cells:
            for rhat in (("none", "last") if op == "nr" else ("none",)):
                for c in (0.0, 0.5, 1.0, 1.5):
                    for ep in range(a.episodes):
                        ret = run(env, team, n, op, "R" if src.startswith("L") else src, phi,
                                  al, ep, rhat, c, adv)
                        w.writerow([a.label, op, src, phi, al, rhat, c, ep, round(ret, 4)])
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
