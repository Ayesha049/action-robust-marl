"""Phase 2 Stage 2 confirmation tests F1-F7 (pre-registered in phase-2/PREREG-STAGE2.md).

Inputs per task: --dirs with the MADDPG base's Stage 0 rows and the trained teams' evaluation rows
(phase2/stage0_faults.py CSVs, predictor none), and --diag, the per-k correction-fit CSV of
phase2/diag_correction.py for RC and FT on rot 90 deg, all agents (F5a).

rho is on the base scale (phase2/stage1_summary.paired_table). delta = 0.05 rho, which is 5% of
the clean-to-random gap. CIs: 95% percentile bootstrap over seeds, 10,000 resamples, seed 0.

  F1 headline   rot 90, all agents: RC - MADDPG >= delta and CI lower > 0
  F2 breadth    rot {30, 60, 90, 180} x {all, worst1}: CI lower of RC - MADDPG > 0 in at least
                5 of the 8 cells, and CI upper < 0 in none
  F3 split      per seed, mean of RC - REC over the 8 F2 cells: CI lower > 0
  F4 dose       per seed, OLS slope of (rho_RC - rho_MADDPG) on theta in {30, 60, 90, 180},
                all agents: CI lower > 0
  F5 mechanism  (a) per seed, mean over k = 1..12 of k_fit(RC) - k_fit(FT), rot 90 all: CI lower > 0
                (b) RC - RC-blind on the F1 cell: CI lower > 0
  F6 no harm    every gain, bias, drift, stuck and wdir cell: CI lower of RC - MADDPG > -delta
  F7 clean      clean return (not rho) RC - MADDPG: CI lower > -0.5
Outcome: Full = all pass; Without split = all but F3 pass; Mechanism only = F5, F6, F7 pass;
No = anything else.
"""

import argparse

import numpy as np
import pandas as pd

from phase2.stage1_summary import boot, paired_table

DELTA = 0.05
F2_CELLS = [("rot", th, sc) for th in (30.0, 60.0, 90.0, 180.0) for sc in ("all", "worst1")]


def diff(w, x, y, cell):
    g = w[(w.kind == cell[0]) & (w.strength == cell[1]) & (w.scope == cell[2])]
    d = (g[x] - g[y]).values
    lo, hi = boot(d)
    return d.mean(), lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--diag", required=True)
    a = p.parse_args()
    t = paired_table(a.dirs, a.task)
    w = t.pivot_table(index=["seed", "kind", "strength", "scope"], columns="method",
                      values="rho").reset_index()
    wr = t.pivot_table(index=["seed", "kind", "strength", "scope"], columns="method",
                       values="ret").reset_index()
    print(f"== {a.task}: {w.seed.nunique()} seeds {sorted(w.seed.unique())}, methods "
          f"{[c for c in w.columns if c not in ('seed', 'kind', 'strength', 'scope')]}")
    res = {}

    m, lo, hi = diff(w, "residual", "maddpg", ("rot", 90.0, "all"))
    res["F1"] = m >= DELTA and lo > 0
    print(f"F1 headline  rot 90 all  RC-MADDPG {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {res['F1']}")

    better = worse = 0
    for c in F2_CELLS:
        m, lo, hi = diff(w, "residual", "maddpg", c)
        better += lo > 0
        worse += hi < 0
        print(f"   F2 cell {c[0]} {c[1]:>5} {c[2]:<6} RC-MADDPG {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    res["F2"] = better >= 5 and worse == 0
    print(f"F2 breadth   better in {better}/8, worse in {worse}  -> {res['F2']}")

    sel = w[w[["kind", "strength", "scope"]].apply(tuple, axis=1).isin(F2_CELLS)]
    per_seed = (sel.residual - sel.rec).groupby(sel.seed).mean()
    lo, hi = boot(per_seed.values)
    res["F3"] = lo > 0
    print(f"F3 split     mean RC-REC over F2 cells {per_seed.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {res['F3']}")

    slopes = []
    for s, g in w[(w.kind == "rot") & (w.scope == "all")].groupby("seed"):
        slopes.append(np.polyfit(g.strength.values, (g.residual - g.maddpg).values, 1)[0])
    lo, hi = boot(slopes)
    res["F4"] = lo > 0
    print(f"F4 dose      slope of RC-MADDPG on theta (per 100 deg) {100 * np.mean(slopes):+.3f} "
          f"[{100 * lo:+.3f}, {100 * hi:+.3f}]  -> {res['F4']}")

    dg = pd.read_csv(a.diag)
    dg = dg[(dg.task == a.task) & (dg.k >= 1) & (dg.k <= 12)]
    kf = dg.pivot_table(index="seed", columns="method", values="k_fit", aggfunc="mean")
    d5 = (kf["residual"] - kf["ft"]).values
    lo, hi = boot(d5)
    f5a = lo > 0
    m, lo2, hi2 = diff(w, "residual", "blind", ("rot", 90.0, "all"))
    f5b = lo2 > 0
    res["F5"] = f5a and f5b
    print(f"F5 mechanism (a) k_fit RC-FT {d5.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {f5a};  "
          f"(b) RC-blind {m:+.3f} [{lo2:+.3f}, {hi2:+.3f}] -> {f5b}  -> {res['F5']}")

    harm = []
    for key, g in w[w.kind.isin(["gain", "bias", "drift", "stuck", "wdir"])].groupby(
            ["kind", "strength", "scope"]):
        d = (g.residual - g.maddpg).values
        lo, hi = boot(d)
        harm.append((key, d.mean(), lo))
    worst = min(harm, key=lambda x: x[2])
    res["F6"] = all(lo > -DELTA for _, _, lo in harm)
    print(f"F6 no harm   {len(harm)} cells; lowest CI bound {worst[2]:+.3f} at {worst[0]} "
          f"(mean {worst[1]:+.3f})  -> {res['F6']}")

    g = wr[wr.kind == "clean"]
    d = (g.residual - g.maddpg).values
    lo, hi = boot(d)
    res["F7"] = lo > -0.5
    print(f"F7 clean     return RC-MADDPG {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {res['F7']}")

    if all(res.values()):
        out = "Full"
    elif all(v for k, v in res.items() if k != "F3"):
        out = "Without split"
    elif res["F5"] and res["F6"] and res["F7"]:
        out = "Mechanism only"
    else:
        out = "No"
    print(f"OUTCOME ({a.task}): {out}   " + " ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in res.items()))


if __name__ == "__main__":
    main()
