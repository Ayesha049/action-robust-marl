"""Summarise Phase 1 Stage 1 and apply the go rule fixed in PHASE-1.md.

Competence retained rho = (J - J_rand) / (J0 - J_rand), J0 = the team's own clean return on the
evaluation seeds, J_rand = -26.08 (spread). Learned cells: mean over the two adversary seeds.
Teams: residual_s*, ft_s* (runs_phase1/), maddpg_s*, nr_s* (Phase 0 runs/). Paired by seed.

Go rule:
 (1) RC - FT averaged over alpha 0.1/0.3/0.5 at phi 0.99: 95% bootstrap CI over seeds excludes 0
 (2) RC - FT point estimate at alpha 0.3 increases R -> phi 0.9 -> phi 0.99
 (3) |mean clean return RC - MADDPG| <= 0.5
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

J_RAND = -26.08


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs_phase1")
    p.add_argument("--runs", default="runs")
    a = p.parse_args()
    ev = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(a.out, "eval", "*.csv")))
    ev["kind"] = ev.label.str.rsplit("_s", n=1).str[0]
    ev["seed"] = ev.label.str.rsplit("_s", n=1).str[1].astype(int)
    m = ev.groupby(["kind", "seed", "op", "source", "phi", "alpha"])["return"].mean().reset_index()
    j0 = m[m.source == "clean"].set_index(["kind", "seed"])["return"].rename("J0")
    m["cell"] = m.op + "-" + m.source + m.phi.map(lambda x: f"{x:g}" if x else "") \
        + "-a" + m.alpha.map(lambda x: f"{x:g}")
    rows = [m[["kind", "seed", "cell", "return"]]]
    adv = []
    for kind, root in (("residual", a.out), ("ft", a.out), ("maddpg", a.runs), ("nr", a.runs)):
        pat = (f"{root}/{kind}_s*/adv/*.json" if root == a.out
               else f"{root}/spread__{kind}__s*/adv/*.json")
        for f in glob.glob(pat):
            d = json.load(open(f))
            name = os.path.basename(f)
            if not (name.startswith("nr_a0.3_all") or name.startswith("pr_a1.0_one")):
                continue
            seed = int(f.split("/")[-3].rsplit("s", 1)[1])
            adv.append({"kind": kind, "seed": seed, "cell": "L-" + name.rsplit("_s", 1)[0],
                        "return": d["mean_return"]})
    if adv:
        adv = pd.DataFrame(adv)
        adv = adv[adv.seed.isin(m.seed.unique())]
        rows.append(adv.groupby(["kind", "seed", "cell"])["return"].mean().reset_index())
    r = pd.concat(rows).join(j0, on=["kind", "seed"])
    r["rho"] = (r["return"] - J_RAND) / (r.J0 - J_RAND)
    r.to_csv(os.path.join(a.out, "stage1_per_team.csv"), index=False)

    w = r.pivot_table(index=["cell", "seed"], columns="kind", values="rho")
    wr = r.pivot_table(index=["cell", "seed"], columns="kind", values="return")
    out = []
    for cell, g in w.groupby(level=0):
        row = {"cell": cell}
        for k in ("maddpg", "nr", "ft", "residual"):
            if k in g:
                row[k] = g[k].mean()
        for k in ("ft", "maddpg", "nr"):
            if k in g and "residual" in g:
                d = (g["residual"] - g[k]).dropna()
                lo, hi = boot(d.values)
                row[f"RC-{k}"], row[f"lo_{k}"], row[f"hi_{k}"] = d.mean(), lo, hi
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, "stage1_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("clean return:", wr.loc["none-clean-a0"].mean().round(3).to_dict())
    print(s.round(3).to_string(index=False))

    # ---- go rule
    rc_ft = (w["residual"] - w["ft"]).rename("d").reset_index()
    phi99 = rc_ft[rc_ft.cell.str.startswith("nr-Rphi0.99")].groupby("seed").d.mean()
    lo, hi = boot(phi99.values)
    g1 = lo > 0
    pts = [rc_ft[rc_ft.cell == c].d.mean() for c in ("nr-R-a0.3", "nr-Rphi0.9-a0.3",
                                                       "nr-Rphi0.99-a0.3")]
    g2 = pts[0] < pts[1] < pts[2]
    clean = wr.loc["none-clean-a0"]
    dc = clean["residual"].mean() - clean["maddpg"].mean()
    g3 = abs(dc) <= 0.5
    print(f"\nGO RULE (1) RC-FT mean over alpha at phi 0.99 = {phi99.mean():+.4f} "
          f"[{lo:.4f}, {hi:.4f}] -> {'pass' if g1 else 'fail'}")
    print(f"GO RULE (2) RC-FT at alpha 0.3: R {pts[0]:+.4f}, phi0.9 {pts[1]:+.4f}, "
          f"phi0.99 {pts[2]:+.4f} -> {'pass' if g2 else 'fail'}")
    print(f"GO RULE (3) clean RC - MADDPG = {dc:+.3f} -> {'pass' if g3 else 'fail'}")
    print("STAGE 1:", "GO" if (g1 and g2 and g3) else "NO-GO")


if __name__ == "__main__":
    main()
