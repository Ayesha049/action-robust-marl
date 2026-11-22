"""Summarise Phase 2 Stage 1 (development): trained RC / RC-blind / REC / FT against the frozen
MADDPG base, on the Stage 0 fault cells (no oracle).

Inputs: --dirs, any number of directories of stage0_faults.py CSVs (plain or .csv.gz). They must
include the MADDPG base's Stage 0 rows for the same task (same seeds and RNG keys, so every
cell is paired per episode) and the trained teams' evaluation rows.

Scale: rho = (J - J_rand) / (J0_base - J_rand), with J0_base the MADDPG base's clean return for
that seed. One common scale for every method, so differences are in base-competence units and
any clean cost shows up as a clean rho below 1.

Victim choice (worst1, wdir): the MADDPG base's worst victim/direction under no correction,
fixed for every method of that seed (paired comparison; no per-method worst-case search).

Writes <out>/stage1_cells.csv (per task, method, cell: mean rho and paired differences with
95% bootstrap CIs over seeds) and prints the main tables.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

J_RAND = {"spread": -26.08, "tag": 1.05}
COMPARE = [("residual", "maddpg"), ("residual", "rec"), ("residual", "blind"), ("residual", "ft"),
           ("rec", "maddpg")]


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def paired_table(dirs, task):
    """Per (method, seed, kind, strength, scope) rho on the base scale, victim fixed by the base."""
    files = [f for d in dirs for f in glob.glob(os.path.join(d, f"{task}__*.csv*"))]
    df = pd.concat(pd.read_csv(f) for f in files)
    df = df[(df.predictor == "none") & (df.task == task)]
    df["direction"] = df.direction.fillna(-1)
    cell = ["kind", "strength", "scope", "direction"]
    m = df.groupby(["method", "seed"] + cell)["return"].mean().reset_index()
    j0 = m[(m.method == "maddpg") & (m.kind == "clean")].set_index("seed")["return"]
    jr = J_RAND[task]
    m["rho"] = (m["return"] - jr) / (m.seed.map(j0) - jr)
    rows = []
    base = m[m.method == "maddpg"]
    for (seed, kind, s), g in base.groupby(["seed", "kind", "strength"]):
        fams = {"wdir": g} if kind == "wdir" else {"all": g[g.scope == "all"], "worst1": g[g.scope != "all"]}
        if kind == "clean":
            fams = {"clean": g}
        for fam, h in fams.items():
            if h.empty:
                continue
            pick = h.loc[h.rho.idxmin()]
            sel = m[(m.seed == seed) & (m.kind == kind) & (m.strength == s) &
                    (m.scope == pick.scope) & (m.direction == pick.direction)]
            for _, r in sel.iterrows():
                rows.append(dict(method=r.method, seed=seed, kind=kind, strength=s, scope=fam,
                                 rho=r.rho, ret=r["return"], J0=j0[seed]))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True)
    a = p.parse_args()
    t = paired_table(a.dirs, a.task)
    methods = sorted(t.method.unique())
    w = t.pivot_table(index=["seed", "kind", "strength", "scope"], columns="method",
                      values="rho").reset_index()
    out = []
    for key, g in w.groupby(["kind", "strength", "scope"]):
        r = dict(zip(["kind", "strength", "scope"], key), n=len(g))
        for meth in methods:
            if meth in g:
                r[meth] = g[meth].mean()
        for x, y in COMPARE:
            if x in g and y in g:
                d = (g[x] - g[y]).values
                r[f"{x}-{y}"] = np.nanmean(d)
                r[f"{x}-{y}_lo"], r[f"{x}-{y}_hi"] = boot(d)
        out.append(r)
    res = pd.DataFrame(out)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, f"stage1_cells_{a.task}.csv"), index=False)
    pd.set_option("display.width", 250)
    show = ["kind", "strength", "scope"] + [x for x in ("maddpg", "residual", "rec", "blind", "ft")
                                          if x in res]
    print(f"== {a.task}: rho on the base scale (mean over {res.n.max()} seeds)")
    print(res[show].round(3).to_string(index=False))
    for x, y in COMPARE:
        c = f"{x}-{y}"
        if c in res:
            print(f"\n== {a.task}: {c} (mean, 95% bootstrap CI over seeds)")
            print(res[["kind", "strength", "scope", c, c + "_lo", c + "_hi"]].round(3)
                  .to_string(index=False))


if __name__ == "__main__":
    main()
