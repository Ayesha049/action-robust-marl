"""Summarise the MIR3 rendezvous reproduction from the run logs
(logs/<token>_train_<method>_s<seed>.log, logs/<token>_adv_<method>_s<seed>_a<agent>.log).

Each log prints "Eval for 32 episodes at timestep T" followed by "Mean of return: R".
Clean return = last eval of the training log; attacked return = last eval of each adversary log.
Writes evals.csv (every eval point) and summary.csv; prints MIR3 - MADDPG with a 95% bootstrap CI
over seeds (attacked return first averaged over the 10 victim agents of each seed).
Teams whose final clean return is not above --min-clean (-40) are reported but excluded from the
attacked comparison, the same rule for both methods.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

PAT = re.compile(r"Eval for \d+ episodes at timestep (\d+)\s*\nMean of return: (-?[\d.]+)")


def evals(path):
    return [(int(t), float(r)) for t, r in PAT.findall(open(path).read())]


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    p.add_argument("--token", default="paper")
    p.add_argument("--min-clean", type=float, default=-40.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    a.out = a.out or f"summary_{a.token}"
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in glob.glob(os.path.join(a.logs, f"{a.token}_train_*.log")):
        m, s = re.match(rf"{a.token}_train_(\w+?)_s(\d+)\.log", os.path.basename(f)).groups()
        rows += [dict(kind="clean", method=m, seed=int(s), agent=-1, t=t, ret=r) for t, r in evals(f)]
    for f in glob.glob(os.path.join(a.logs, f"{a.token}_adv_*.log")):
        m, s, ag = re.match(rf"{a.token}_adv_(\w+?)_s(\d+)_a(\d+)\.log", os.path.basename(f)).groups()
        rows += [dict(kind="attacked", method=m, seed=int(s), agent=int(ag), t=t, ret=r) for t, r in evals(f)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "evals.csv"), index=False)
    last = df.sort_values("t").groupby(["kind", "method", "seed", "agent"]).tail(1)
    per_seed = last.groupby(["kind", "method", "seed"]).ret.mean().unstack("kind")
    per_seed["worst_agent"] = last[last.kind == "attacked"].groupby(["method", "seed"]).ret.min()
    per_seed["learned"] = per_seed.clean > a.min_clean
    per_seed.to_csv(os.path.join(a.out, "summary.csv"))
    print(per_seed.round(2).to_string())
    print(f"learned (clean > {a.min_clean}):", per_seed.groupby("method").learned.sum().to_dict())
    print(per_seed.groupby("method").agg(["mean", "std", "count"]).round(2).to_string())
    for col in ["clean", "attacked", "worst_agent"]:
        if col not in per_seed:
            continue
        src = per_seed if col == "clean" else per_seed[per_seed.learned]
        w = src[col].unstack("method").dropna()
        if {"maddpg", "mir3"} <= set(w.columns) and len(w) >= 2:
            d = w.mir3 - w.maddpg
            lo, hi = boot_ci(d.values)
            print(f"{col}: MIR3 - MADDPG = {d.mean():+.2f} [{lo:.2f}, {hi:.2f}] (n={len(d)} paired seeds)")
        for m in ("maddpg", "mir3"):
            if m in src[col].unstack("method"):
                v = src[col].unstack("method")[m].dropna()
                if len(v) >= 2:
                    lo, hi = boot_ci(v.values)
                    print(f"  {col} {m}: {v.mean():.2f} [{lo:.2f}, {hi:.2f}] (n={len(v)})")


if __name__ == "__main__":
    main()
