"""Phase 0b results on the reduced ("core") attack grid.

Core grid (competence retained only):
  R (random)  : PR, NR  x  scope all / worst single agent  x  alpha 0.1, 0.3, 0.5
  L (learned) : PR, NR  x  scope all / one agent (agent 0) x  alpha 0.1, 0.3
                PR takeover (alpha 1, one agent)
The gradient source, the mean-over-victims scope, the 11-point curves, AUCs and
breakdown alpha are reported only in the appendix (aggregate.py outputs).

Reads curves.csv and learned.csv written by aggregate.py. Writes to --out:
  core_cells.csv      mean retained per (task, method, cell) and clean return
  core_verdicts.csv   method - MADDPG, paired by seed, 95% bootstrap CI
  core_rank.csv       Kendall tau between method rankings for every pair of cells
  core_r_predicts_l.csv  Spearman rho across methods, R vs L at matched op/alpha/scope
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

R_ALPHAS = (0.1, 0.3, 0.5)
L_ALPHAS = (0.1, 0.3)


def boot_ci(x, n=10_000, seed=0):  # same as aggregate.py
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def core_long(curves, learned, exclude):
    c = curves[~curves.method.isin(exclude)]
    one = c[c.scope.str.startswith("one")]
    worst = (one.groupby(["task", "method", "seed", "source", "op", "alpha"]).retained.min()
             .reset_index().assign(scope="worst1"))
    r = pd.concat([c[c.scope.isin(["all", "one0"])], worst])
    r = r[(r.source == "R") & r.alpha.round(2).isin(R_ALPHAS)]
    r = r.assign(cell="R-" + r.op + "-" + r.scope + "-a" + r.alpha.round(2).astype(str))
    lr = learned[~learned.method.isin(exclude)]
    lr = lr[lr.alpha.round(2).isin(L_ALPHAS + (1.0,))]
    lr = lr.assign(cell="L-" + lr.op + "-" + lr.scope + "-a" + lr.alpha.round(2).astype(str))
    keep = ["task", "method", "seed", "cell", "retained"]
    return pd.concat([r[keep], lr[keep]], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves", required=True)
    p.add_argument("--learned", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", nargs="*", default=["marta"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    curves, learned = pd.read_csv(a.curves), pd.read_csv(a.learned)
    df = core_long(curves, learned, a.exclude)
    main_cells = df[~df.cell.str.contains("one0")]   # R one0 is only for the R-vs-L check
    clean = (curves[~curves.method.isin(a.exclude)].groupby(["task", "method", "seed"]).clean
             .first().reset_index())

    tab = main_cells.groupby(["task", "method", "cell"]).retained.mean().unstack("cell")
    tab = tab.join(clean.groupby(["task", "method"]).clean.mean())
    tab.round(4).to_csv(os.path.join(a.out, "core_cells.csv"))

    rows = []
    long = pd.concat([main_cells.rename(columns={"retained": "value"}).assign(metric="retained"),
                      clean.rename(columns={"clean": "value"}).assign(cell="clean", metric="clean")])
    for (t, cell), g in long.groupby(["task", "cell"]):
        base = g[g.method == "maddpg"].set_index("seed").value
        for m in sorted(set(g.method) - {"maddpg"}):
            d = (g[g.method == m].set_index("seed").value - base).dropna()
            lo, hi = boot_ci(d.values)
            v = "better" if lo > 0 else "worse" if hi < 0 else "unclear"
            rows.append(dict(task=t, cell=cell, method=m, n=len(d), mean_diff=d.mean(),
                             ci_low=lo, ci_high=hi, verdict=v))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.out, "core_verdicts.csv"), index=False)

    rk = []
    for t, g in main_cells.groupby("task"):
        m = g.groupby(["cell", "method"]).retained.mean().unstack("method")
        for c1, c2 in combinations(m.index, 2):
            tau, pv = kendalltau(m.loc[c1], m.loc[c2])
            rk.append(dict(task=t, cell_a=c1, cell_b=c2, kendall_tau=tau, p=pv))
    pd.DataFrame(rk).round(4).to_csv(os.path.join(a.out, "core_rank.csv"), index=False)

    rl = []
    for t, g in df.groupby("task"):
        m = g.groupby(["cell", "method"]).retained.mean().unstack("method")
        for op in ("pr", "nr"):
            for al in L_ALPHAS:
                for rs, ls in (("all", "all"), ("one0", "one")):
                    rc, lc = f"R-{op}-{rs}-a{al}", f"L-{op}-{ls}-a{al}"
                    rho, pv = spearmanr(m.loc[rc], m.loc[lc])
                    rl.append(dict(task=t, r_cell=rc, l_cell=lc, spearman_rho=rho, p=pv))
    pd.DataFrame(rl).round(4).to_csv(os.path.join(a.out, "core_r_predicts_l.csv"), index=False)
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
