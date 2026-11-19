"""Phase 1 confirmation run: revised mechanism test T3' (pre-registered in PHASE-1.md).

Cell: L-NR all agents, alpha 0.3, each team's own learned adversaries (seeds 0, 1), Phase 0
evaluation episodes (100_000 + ep, 100 per adversary).
(a) Compensation: per team seed, fit delta = cmd - mu_b ~ k_amp * mu_b + k_comp * (-r_{t-1})
    (least squares pooled over agents, both components, steps t >= 1, both adversaries;
    mu_b = frozen PR-MADDPG base command at the same observation). Passes if
    k_comp(RC) - k_comp(FT), paired by seed, has a 95% bootstrap CI above 0.
(b) Evidence is causal: RC-blind = RC with prev_exe replaced by prev_cmd (own velocity in the
    observation still carries the executed action). Passes if rho(RC) - rho(RC-blind) on RC's
    own adversaries, paired by seed, has a 95% bootstrap CI above 0. rho uses RC's clean return
    (RC-blind is identical on clean episodes).
T3' passes if (a) and (b) both pass. Writes <out>/t3prime_per_seed.csv and t3prime.txt.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from marl.envs import make_env
from phase1.diag_stage1 import load_adv, run
from phase1.residual import load_base, load_team_any

J_RAND = -26.08


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def k_comp(rows):
    d = np.array(rows, float)  # columns as in diag_stage1.run
    t, cmd, mub, rprev = d[:, 1], d[:, 3:5], d[:, 5:7], d[:, 11:13]
    k = t >= 1
    y = (cmd - mub)[k].T.reshape(-1)
    X = np.c_[mub[k].T.reshape(-1), -rprev[k].T.reshape(-1)]
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


def episodes(env, team, bases, n, adv, blind, eps):
    rows, rets = [], []
    for ep in range(eps):
        r = run(env, team, bases, n, "nr", "L", 0.0, 0.3, ep, adv, blind)
        rows += r
        rets.append(sum(x[-1] for x in r if x[2] == 0))
    return rows, float(np.mean(rets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs_phase1c")
    p.add_argument("--seeds", default="5 6 7 8 9")
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(1)
    env = make_env("spread")
    n = env.n_team
    od = sum(env.obs_dims[:n])
    res = []
    for s in map(int, a.seeds.split()):
        bases = [load_base(f"runs/spread__pr__s{s}/actor_{i}.pt") for i in range(n)]
        row = {"seed": s}
        for kind in ("residual", "ft"):
            d = os.path.join(a.out, f"{kind}_s{s}")
            team = load_team_any(d, n)
            advs = [load_adv(os.path.join(d, "adv", f"nr_a0.3_all_s{k}.pt"), od, 2 * n) for k in (0, 1)]
            rows, rets = [], []
            for adv in advs:
                rw, rt = episodes(env, team, bases, n, adv, False, a.episodes)
                rows += rw
                rets.append(rt)
            row[f"k_comp_{kind}"] = k_comp(rows)
            row[f"ret_{kind}"] = float(np.mean(rets))
            if kind == "residual":
                clean = [sum(x[-1] for x in run(env, team, bases, n, "none", "clean", 0.0, 0.0, ep)
                             if x[2] == 0) for ep in range(a.episodes)]
                j0 = float(np.mean(clean))
                bl = [episodes(env, team, bases, n, adv, True, a.episodes)[1] for adv in advs]
                row["J0_residual"], row["ret_blind"] = j0, float(np.mean(bl))
                row["rho_residual"] = (row["ret_residual"] - J_RAND) / (j0 - J_RAND)
                row["rho_blind"] = (row["ret_blind"] - J_RAND) / (j0 - J_RAND)
        res.append(row)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(a.out, "t3prime_per_seed.csv"), index=False)
    da = df.k_comp_residual - df.k_comp_ft
    db = df.rho_residual - df.rho_blind
    (la, ha), (lb, hb) = boot(da), boot(db)
    pa, pb = la > 0, lb > 0
    txt = "\n".join([
        f"(a) k_comp RC {df.k_comp_residual.mean():+.4f}, FT {df.k_comp_ft.mean():+.4f}; "
        f"RC - FT {da.mean():+.4f} [{la:+.4f}, {ha:+.4f}] -> {'pass' if pa else 'fail'}",
        f"(b) rho RC {df.rho_residual.mean():.4f}, RC-blind {df.rho_blind.mean():.4f}; "
        f"RC - blind {db.mean():+.4f} [{lb:+.4f}, {hb:+.4f}] -> {'pass' if pb else 'fail'}",
        f"T3': {'pass' if pa and pb else 'fail'}"])
    print(df.round(4).to_string(index=False))
    print(txt)
    open(os.path.join(a.out, "t3prime.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
