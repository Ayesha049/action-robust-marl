"""Summarise Phase 1 Stage 0 (stage0_checks.py outputs).

Competence retained rho = (J - J_rand) / (J0 - J_rand), J0 = the team's clean return on the same
evaluation seeds, J_rand = -26.08 on spread (Phase 0). Learned (L) cells are averaged over the two
adversary seeds first (Phase 0 rule). Gains are oracle predictor minus "none", per team, then mean
over the 5 seeds with a 95% bootstrap CI.

Stop rule (PHASE-1.md, Stage 0): stop if the "last" oracle gains < 0.02 over "none" in every NR
cell for MADDPG.

Also checks that the "none" L cells reproduce Phase 0's adversary evaluation (<run>/adv/*.json).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

J_RAND = {"spread": -26.08, "tag": 1.05}


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="phase1/stage0_out")
    p.add_argument("--runs", default="runs")
    p.add_argument("--threshold", type=float, default=0.02)
    a = p.parse_args()
    df = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(a.dir, "*.csv")))
    team = ["task", "method", "seed"]
    j0 = df[df.source == "clean"].groupby(team)["return"].mean().rename("J0")
    cell = ["source", "phi", "alpha", "scope"]
    m = df[df.source != "clean"].groupby(team + cell + ["adv_seed", "predictor"])["return"].mean()
    m = m.groupby(team + cell + ["predictor"]).mean().reset_index()  # average adversary seeds
    m = m.join(j0, on=team)
    m["rho"] = (m["return"] - m.task.map(J_RAND)) / (m.J0 - m.task.map(J_RAND))
    m.to_csv(os.path.join(a.dir, "summary_per_team.csv"), index=False)

    w = m.pivot_table(index=team + cell, columns="predictor", values="rho").reset_index()
    rows = []
    for key, g in w.groupby(["task", "method"] + cell):
        r = dict(zip(["task", "method"] + cell, key))
        r["n"] = len(g)
        r["rho_none"] = g["none"].mean()
        for pred in ("zero", "last", "exact"):
            d = g[pred] - g["none"]
            lo, hi = boot(d.values)
            r[f"gain_{pred}"], r[f"lo_{pred}"], r[f"hi_{pred}"] = d.mean(), lo, hi
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.dir, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("clean J0 by method:", j0.groupby("method").mean().round(2).to_dict())
    cols = ["method"] + cell + ["rho_none", "gain_zero", "gain_last", "lo_last", "hi_last",
                                "gain_exact"]
    print(s[cols].round(3).to_string(index=False))

    mad = s[s.method == "maddpg"]
    go = (mad.gain_last >= a.threshold).any()
    print(f"\nSTOP RULE: max MADDPG 'last' gain = {mad.gain_last.max():.3f} "
          f"(threshold {a.threshold}) -> {'GO' if go else 'STOP'}")

    sat, spd = [], []
    for f in glob.glob(os.path.join(a.dir, "*.json")):
        d = json.load(open(f))
        for k, v in d["saturation"].items():
            sat.append({"method": d["method"], "seed": d["seed"], "cell": k, **v})
        for k, v in d["speed"].items():
            spd.append({"method": d["method"], "seed": d["seed"], "cell": k, **v})
    sat = pd.DataFrame(sat).groupby(["method", "cell"]).mean(numeric_only=True).drop(columns="seed")
    spd = pd.DataFrame(spd)
    spd["cell"] = spd.cell.str.replace(r"_adv\d", "_adv", regex=True)
    spd = spd.groupby(["method", "cell"]).mean(numeric_only=True).drop(columns="seed")
    sat.to_csv(os.path.join(a.dir, "saturation.csv"))
    spd.to_csv(os.path.join(a.dir, "speed.csv"))
    print("\nSaturation (share of |a| <= threshold):\n", sat.round(3).to_string())
    print("\nReplacement-action speed:\n", spd.round(3).to_string())

    chk = []
    for f in glob.glob(os.path.join(a.runs, "*", "adv", "nr_*.json")):
        d = json.load(open(f))
        if d["task"] != "spread":
            continue
        sel = df[(df.method == d["method"]) & (df.seed == d["seed"]) & (df.source == "L")
                 & (df.alpha == d["alpha"]) & (df.scope == d["scope"])
                 & (df.adv_seed == d["adv_seed"]) & (df.predictor == "none")]
        if len(sel):
            chk.append(abs(sel["return"].mean() - d["mean_return"]))
    if chk:
        print(f"\nPhase 0 reproduction (L, none): {len(chk)} adversaries, "
              f"max |difference| in mean return = {max(chk):.4f}")


if __name__ == "__main__":
    main()
