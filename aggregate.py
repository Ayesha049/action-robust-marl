"""Aggregate Phase 0 MARL results into tables, verdicts and figures.

Competence retained = (attacked - random_team) / (clean - random_team), where
random_team is the return of a team acting uniformly at random (same seeds).
1 = unaffected, 0 = no better than random. This works with negative returns
(simple_spread) where ratios to the clean return would not.

Outputs (in --out):
  baseline_random.csv  random-team return per task
  curves.csv           mean return per (task, method, seed, source, op, scope, alpha)
  metrics.csv          per seed and cell family: clean, auc, retained_auc, breakdown,
                       worst_victim_gap
  learned.csv          learned-adversary returns (+ retained)
  verdicts.csv         method minus MADDPG, paired by seed, 95% bootstrap CI
  rank_stability.csv   Kendall tau between method rankings of every pair of cells
  q4_correlation.csv   R-curve robustness vs learned-attack robustness across methods
  fig_*.png
"""

import argparse
import glob
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

METHODS = ["maddpg", "m3ddpg", "ernie", "mir3", "marta", "pr", "nr"]
LABEL = {"maddpg": "MADDPG", "m3ddpg": "M3DDPG", "ernie": "ERNIE-A-c", "mir3": "MIR3",
         "marta": "MARTA", "pr": "PR-MADDPG", "nr": "NR-MADDPG"}
COLOR = {"maddpg": "#4c4c4c", "m3ddpg": "#8c6bb1", "ernie": "#2ca02c", "mir3": "#e7298a",
         "marta": "#e6ab02", "pr": "#1f77b4", "nr": "#d95f02"}


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def random_team_return(task, episodes=200):
    from marl.envs import make_env
    env = make_env(task)
    rng = np.random.default_rng(0)
    out = []
    for ep in range(episodes):
        env.reset(seed=100_000 + ep)
        tot, done = 0.0, False
        while not done:
            _, _, r, done = env.step(rng.uniform(-1, 1, (env.n, 2)).astype(np.float32))
            tot += r
        out.append(tot)
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs"], help="one or more run roots")
    p.add_argument("--out", default="results/phase0")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(f for r in a.runs for f in glob.glob(os.path.join(r, "*", "eval_curves.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    shas = sorted({json.load(open(os.path.join(os.path.dirname(f), "config.json")))["git_sha"]
                   for f in files})

    rand = {t: random_team_return(t) for t in df.task.unique()}
    pd.Series(rand, name="random_team_return").to_csv(os.path.join(a.out, "baseline_random.csv"))

    keys = ["task", "method", "seed", "source", "op", "scope", "alpha"]
    curves = df.groupby(keys)["return"].mean().reset_index()
    clean = (curves[curves.alpha == 0].groupby(["task", "method", "seed"])["return"].mean()
             .rename("clean"))
    curves = curves.join(clean, on=["task", "method", "seed"])
    curves["random"] = curves.task.map(rand)
    curves["retained"] = (curves["return"] - curves.random) / (curves.clean - curves.random)
    curves.to_csv(os.path.join(a.out, "curves.csv"), index=False)

    # scope families: "one" = mean over single-agent victims, "worst1" = worst victim, "all"
    one = curves[curves.scope.str.startswith("one")]
    fam = pd.concat([
        one.groupby(["task", "method", "seed", "source", "op", "alpha"])[["return", "retained"]]
           .mean().reset_index().assign(scope="one"),
        one.groupby(["task", "method", "seed", "source", "op", "alpha"])[["return", "retained"]]
           .min().reset_index().assign(scope="worst1"),
        curves[curves.scope == "all"][["task", "method", "seed", "source", "op", "alpha",
                                       "return", "retained"]].assign(scope="all"),
    ])
    rows = []
    for (t, m, s, src, op, sc), g in fam.groupby(["task", "method", "seed", "source", "op", "scope"]):
        g = g.sort_values("alpha")
        below = g[g.retained < 0.5].alpha
        rows.append(dict(task=t, method=m, seed=s, source=src, op=op, scope=sc,
                         clean=clean[(t, m, s)], auc=g["return"].mean(),
                         retained_auc=g.retained.mean(),
                         breakdown=below.min() if len(below) else np.inf))
    metrics = pd.DataFrame(rows)
    gap = (metrics[metrics.scope == "one"].set_index(["task", "method", "seed", "source", "op"]).auc
           - metrics[metrics.scope == "worst1"].set_index(["task", "method", "seed", "source", "op"]).auc)
    metrics = metrics.join(gap.rename("worst_victim_gap"), on=["task", "method", "seed", "source", "op"])
    metrics.to_csv(os.path.join(a.out, "metrics.csv"), index=False)

    # learned adversaries
    lj = [json.load(open(f)) for r in a.runs for f in glob.glob(os.path.join(r, "*", "adv", "*.json"))]
    learned = pd.DataFrame([{k: v for k, v in d.items() if k != "returns"} for d in lj])
    if len(learned):
        learned = (learned.groupby(["task", "method", "seed", "op", "alpha", "scope"])["mean_return"]
                   .mean().reset_index().rename(columns={"mean_return": "return"}))
        learned = learned.join(clean, on=["task", "method", "seed"])
        learned["retained"] = ((learned["return"] - learned.task.map(rand))
                               / (learned.clean - learned.task.map(rand)))
        learned.to_csv(os.path.join(a.out, "learned.csv"), index=False)

    # verdicts: method - maddpg, paired by seed
    cells = []
    for (t, src, op, sc), g in metrics.groupby(["task", "source", "op", "scope"]):
        cells.append((f"{src}-{op}-{sc}", t, g, ["clean", "auc", "retained_auc"]))
    if len(learned):
        for (t, op, al, sc), g in learned.groupby(["task", "op", "alpha", "scope"]):
            cells.append((f"L-{op}-a{al}-{sc}", t, g.rename(columns={"return": "auc",
                                                                       "retained": "retained_auc"}),
                          ["auc", "retained_auc"]))
    ver = []
    for cell, t, g, mets in cells:
        base = g[g.method == "maddpg"].set_index("seed")
        for m in METHODS[1:]:
            rob = g[g.method == m].set_index("seed")
            seeds = base.index.intersection(rob.index)
            if len(seeds) < 2:
                continue
            for met in mets:
                d = (rob.loc[seeds, met] - base.loc[seeds, met]).values
                lo, hi = boot_ci(d)
                ver.append(dict(task=t, cell=cell, method=m, metric=met, n=len(seeds),
                                mean_diff=d.mean(), ci_low=lo, ci_high=hi,
                                verdict="better" if lo > 0 else "worse" if hi < 0 else "no clear difference"))
    pd.DataFrame(ver).to_csv(os.path.join(a.out, "verdicts.csv"), index=False)

    # rank stability (Q3): method ranking by mean retained robustness in each cell
    rank_tbl = {}
    for cell, t, g, mets in cells:
        r = g.groupby("method")["retained_auc"].mean()
        if r.index.nunique() >= 3:
            rank_tbl[(t, cell)] = r
    rs = []
    for (k1, r1), (k2, r2) in combinations(rank_tbl.items(), 2):
        if k1[0] != k2[0]:
            continue
        common = r1.index.intersection(r2.index)
        tau, pval = kendalltau(r1[common], r2[common])
        rs.append(dict(task=k1[0], cell_a=k1[1], cell_b=k2[1], kendall_tau=tau, p=pval,
                       n_methods=len(common)))
    pd.DataFrame(rs).to_csv(os.path.join(a.out, "rank_stability.csv"), index=False)

    # Q4: does random-attack robustness predict learned-attack robustness?
    if len(learned):
        q4 = []
        for t in learned.task.unique():
            rr = metrics[(metrics.task == t) & (metrics.source == "R")].groupby(
                ["method", "op", "scope"]).retained_auc.mean()
            for (op, al, sc), g in learned[learned.task == t].groupby(["op", "alpha", "scope"]):
                lm = g.groupby("method").retained.mean()
                sc_r = "one" if sc == "one" else "all"
                x = rr.xs((op, sc_r), level=("op", "scope")) if (op, sc_r) in rr.droplevel(0).index else None
                if x is None:
                    continue
                common = x.index.intersection(lm.index)
                if len(common) >= 3:
                    rho, pv = spearmanr(x[common], lm[common])
                    q4.append(dict(task=t, learned_cell=f"{op}-a{al}-{sc}", spearman=rho, p=pv,
                                   n_methods=len(common)))
        pd.DataFrame(q4).to_csv(os.path.join(a.out, "q4_correlation.csv"), index=False)

    # figures: retained competence vs alpha
    for t in fam.task.unique():
        for src in sorted(fam.source.unique()):
            fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
            for r_, op in enumerate(["pr", "nr"]):
                for c_, sc in enumerate(["one", "worst1", "all"]):
                    ax = axes[r_][c_]
                    for m in METHODS:
                        g = fam[(fam.task == t) & (fam.source == src) & (fam.op == op)
                                & (fam.scope == sc) & (fam.method == m)]
                        if g.empty:
                            continue
                        piv = g.pivot(index="seed", columns="alpha", values="retained")
                        mu = piv.mean(axis=0)
                        lo, hi = zip(*[boot_ci(piv[c].values) for c in piv.columns])
                        ax.plot(piv.columns, mu, color=COLOR[m], label=LABEL[m], lw=2)
                        ax.fill_between(piv.columns, lo, hi, color=COLOR[m], alpha=0.15)
                    ax.set_title(f"{op.upper()} | {sc}")
                    ax.axhline(0.5, ls=":", c="grey")
                    ax.grid(alpha=0.3)
                    if c_ == 0:
                        ax.set_ylabel("competence retained")
                    if r_ == 1:
                        ax.set_xlabel("alpha")
            axes[0][0].legend(fontsize=8)
            fig.suptitle(f"{t}: source {src}")
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, f"fig_{t}_{src}.png"), dpi=140)
            plt.close(fig)

    json.dump({"git_shas": shas, "n_runs": len(files), "random_team": rand},
              open(os.path.join(a.out, "provenance.json"), "w"), indent=2)
    print(f"aggregated {len(files)} runs, {len(learned)} learned-adversary cells -> {a.out}")


if __name__ == "__main__":
    main()
