"""Phase 2 Stage 1 diagnostic: does a trained corrector's correction point where the oracle's does?

For a trained team (stage1.pt) under one fault cell, at every step k = t - t0 after onset, for
each victim agent:
  delta  = a_RC(t) - mu_b(o_t)                     the correction actually applied
  target = clip(oracle command for mu_b) - mu_b    the exact oracle's correction (phase2/faults.py)
Reported per k (mean over victims and episodes): cosine(delta, target), the least-squares slope
k_fit of delta on target (1 = full correction, 0 = none), |delta| and |target|.
Also before the onset (k < 0), where target = 0 and |delta| shows false corrections.
Episodes use the Stage 0 evaluation seeds and keys.
"""

import argparse
import json
import os

import numpy as np
import torch

from marl.attack import load_team
from marl.envs import make_env
from phase2.faults import Fault, oracle_command
from phase2.stage0_faults import KIND_ID
from phase2.teams import load_any


def correction_fit(run_dir, kind="rot", strength=90.0, scope="all", episodes=100):
    """Per step k after onset: (cos(delta, target), k_fit, mean |delta|, mean |target|)."""
    a = argparse.Namespace(run_dir=run_dir, kind=kind, strength=strength, scope=scope,
                           episodes=episodes)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    env = make_env(cfg["task"], prey_path="assets/tag_prey.pt")
    n = env.n_team
    team = load_any(a.run_dir, n)
    base = load_team(cfg["base_dir"], n)
    victims = list(range(n)) if a.scope == "all" else [int(a.scope)]
    fault = Fault(a.kind, a.strength, victims, n)
    rows = {}
    for e in range(a.episodes):
        ep = fault.sample(np.random.default_rng([2026, KIND_ID[a.kind], 9, e]))
        obs = env.reset(seed=100_000 + e)
        team.reset()
        done, t = False, 0
        while not done:
            a_rc = team.act(obs)
            mu_b = base(obs)
            cmd_star, _, _ = oracle_command(fault, ep, t, mu_b, "exact")
            exe = fault.execute(ep, t, a_rc)
            team.observe(a_rc, exe)
            k = t - ep["t0"]
            for j in victims:
                rows.setdefault(k, []).append((a_rc[j] - mu_b[j], cmd_star[j] - mu_b[j]))
            obs, _, _, done = env.step(exe)
            t += 1
    out = {}
    for k in sorted(rows):
        d = np.array([r[0] for r in rows[k]])
        g = np.array([r[1] for r in rows[k]])
        nd, ng = np.linalg.norm(d, axis=1), np.linalg.norm(g, axis=1)
        ok = (nd > 1e-6) & (ng > 1e-6)
        cos = np.mean((d[ok] * g[ok]).sum(1) / (nd[ok] * ng[ok])) if ok.any() else np.nan
        kfit = (d * g).sum() / max((g * g).sum(), 1e-9)
        out[k] = (cos, kfit, nd.mean(), ng.mean())
    return cfg, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--kind", default="rot")
    p.add_argument("--strength", type=float, default=90.0)
    p.add_argument("--scope", default="all", help="all, or a victim index")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--csv", default="", help="also append per-k rows to this CSV")
    a = p.parse_args()
    torch.set_num_threads(1)
    cfg, out = correction_fit(a.run_dir, a.kind, a.strength, a.scope, a.episodes)
    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a") as f:
            if new:
                f.write("task,method,seed,kind,strength,scope,k,cos,k_fit,delta,target\n")
            for k, (cos, kfit, nd, ng) in out.items():
                f.write(f"{cfg['task']},{cfg['method']},{cfg['base_seed']},{a.kind},{a.strength},"
                        f"{a.scope},{k},{cos:.4f},{kfit:.4f},{nd:.4f},{ng:.4f}\n")
    print(f"{cfg['task']} {cfg['method']} base seed {cfg['base_seed']}: {a.kind} {a.strength} {a.scope}")
    print("   k  cos(delta,target)  k_fit  |delta|  |target|")
    for k, (cos, kfit, nd, ng) in out.items():
        if -3 <= k <= 12:
            print(f"{k:4d}  {cos:17.2f}  {kfit:5.2f}  {nd:7.2f}  {ng:8.2f}")


if __name__ == "__main__":
    main()
