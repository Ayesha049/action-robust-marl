"""Phase 2 Stage 0: persistent-fault headroom on a Phase 0 checkpoint, no training.

For one frozen team, every fault cell is run for 100 episodes on Phase 0's evaluation seeds
(100_000 + episode), with each oracle predictor of phase2/faults.py (none, exact, d1, d3, d10).
Per episode it writes the return and the share of corrected action components that the oracle
had to clip. That share is the empirical complement of the restorable set: clip(g a*) cannot be
undone where |a*| > g, nor clip(a* + b) where a* - b leaves [-1, 1].

Cells (spread, n = 3):
  clean
  gain   g   in {0.7, 0.5, 0.3}      victims: each single agent, and all
  bias   |b| in {0.1, 0.3, 0.5}      victims: each single agent, and all; random direction
  drift  eta in {0.01, 0.02, 0.04}   victims: each single agent
  stuck                              victims: each single agent (predictor none only)
  wdir   bias |b| 0.3, 8 fixed directions x each single agent (worst-direction grid)
  rot    theta in {30, 60, 90, 180} degrees  victims: each single agent, and all (opt-in kind)

Writes <out>/<label>__<kinds>.csv. Use --kinds to split one team across several jobs.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from marl.envs import make_env
from phase2.faults import ONSET_BY_TASK, ONSET_MAX, Fault, oracle_command
from phase2.teams import load_any

STRENGTHS = {"gain": (0.7, 0.5, 0.3), "bias": (0.1, 0.3, 0.5), "drift": (0.01, 0.02, 0.04),
             "rot": (30.0, 60.0, 90.0, 180.0)}
KIND_ID = {"clean": 0, "gain": 1, "bias": 2, "drift": 3, "stuck": 4, "wdir": 5, "rot": 6}


def cells_for(kind, n):
    """(kind, fault kind, strength, scope, victims, direction) tuples."""
    singles = [(f"a{j}", [j]) for j in range(n)]
    if kind == "clean":
        return [("clean", None, 0.0, "all", [], None)]
    if kind in ("gain", "bias", "rot"):
        return [(kind, kind, s, sc, v, None) for s in STRENGTHS[kind]
                for sc, v in singles + [("all", list(range(n)))]]
    if kind == "drift":
        return [(kind, kind, s, sc, v, None) for s in STRENGTHS[kind] for sc, v in singles]
    if kind == "stuck":
        return [(kind, kind, 1.0, sc, v, None) for sc, v in singles]
    if kind == "wdir":
        return [(kind, "bias", 0.3, sc, v, d) for d in range(8) for sc, v in singles]
    raise ValueError(kind)


def episode(env, team, fault, predictor, ep_seed, rng_key):
    rng = np.random.default_rng(rng_key)
    ep = fault.sample(rng) if fault else None
    obs = env.reset(seed=ep_seed)
    team.reset()
    total, done, t, clipped, corrected = 0.0, False, 0, 0, 0
    while not done:
        a_star = team.act(obs)
        cmd = a_star
        if fault:
            cmd, c, k = oracle_command(fault, ep, t, a_star, predictor)
            clipped, corrected = clipped + c, corrected + k
            exe = fault.execute(ep, t, cmd)
        else:
            exe = a_star
        team.observe(cmd, exe)
        obs, _, team_r, done = env.step(exe)
        total += team_r
        t += 1
    return total, (ep["t0"] if ep else -1), (clipped / corrected if corrected else np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kinds", default="clean,gain,bias,drift,stuck,wdir")
    p.add_argument("--predictors", default="none,exact,d1,d3,d10")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--strengths", default="",
                   help="override, e.g. 'bias=0.7,0.9' (exploratory runs; write to another --out)")
    a = p.parse_args()
    for item in filter(None, a.strengths.split(";")):
        k, v = item.split("=")
        STRENGTHS[k] = tuple(float(x) for x in v.split(","))
    torch.set_num_threads(1)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    env = make_env(cfg["task"])
    n = env.n_team
    team = load_any(a.run_dir, n)  # Phase 0 (stateless) or trained Phase 1/2 (recurrent) team
    seed = cfg.get("base_seed", cfg["seed"])  # trained teams are labelled by their base's seed
    preds = a.predictors.split(",")
    kinds = a.kinds.split(",")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{cfg['task']}__{cfg['method']}__s{seed}__{'-'.join(kinds)}.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "seed", "kind", "strength", "scope", "direction",
                    "predictor", "episode", "onset", "return", "clip_share"])
        for kind in kinds:
            for ci, (label, fk, s, scope, victims, d) in enumerate(cells_for(kind, n)):
                fault = (Fault(fk, s, victims, n, direction=d,
                               onset_max=ONSET_BY_TASK.get(cfg["task"], ONSET_MAX)) if fk else None)
                cell_preds = ["none"] if fk in (None, "stuck") else preds
                for pred in cell_preds:
                    for e in range(a.episodes):
                        # the key depends on the fault kind, direction and episode only: identical
                        # onsets and directions across predictors, methods, strengths and scopes
                        key = [2026, KIND_ID[label], 9 if d is None else d, e]
                        ret, t0, cs = episode(env, team, fault, pred, 100_000 + e, key)
                        w.writerow([cfg["task"], cfg["method"], seed, label, s, scope,
                                    "" if d is None else d, pred, e, t0, round(ret, 4),
                                    "" if np.isnan(cs) else round(cs, 4)])
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
