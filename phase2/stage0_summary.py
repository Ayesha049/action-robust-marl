"""Summarise Phase 2 Stage 0 (phase2/stage0_faults.py outputs).

Competence retained rho = (J - J_rand) / (J0 - J_rand), with J0 the team's clean return on the
same evaluation seeds and J_rand = -26.08 on spread (Phase 0).

Scopes:
  all      every agent faulted
  worst1   Phase 0 rule: per team, the single victim with the lowest rho under "none"; oracle
           predictors are evaluated on that same victim, so a gain is paired with its baseline
  wdir     worst (direction, victim) pair under "none", the same pairing rule

Oracle gain = rho(predictor) - rho(none), per team, then mean over seeds with a 95% bootstrap
CI. clip_share = share of corrected action components the exact oracle had to clip (where full
restoration is impossible).

Stop rule (phase-2/PHASE-2.md, Stage 0; proposed 2026-11-21): stop the spread arm if the exact
oracle on MADDPG gains < 0.05 in every gain and bias cell.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

J_RAND = {"spread": -26.08}
TEAM = ["task", "method", "seed"]


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="phase2/stage0_out")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--phase1-stage0", default="",
                   help="dir of Phase 1 Stage 0 CSVs, to check clean returns reproduce")
    a = p.parse_args()
    df = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(a.dir, "*.csv")))
    j0 = df[df.kind == "clean"].groupby(TEAM)["return"].mean().rename("J0")

    old_files = glob.glob(os.path.join(a.phase1_stage0, "spread__*.csv")) if a.phase1_stage0 else []
    if a.phase1_stage0 and not old_files:
        print(f"clean-return reproduction: no Phase 1 Stage 0 CSVs in {a.phase1_stage0}, skipped")
    if old_files:
        old = pd.concat(pd.read_csv(f) for f in old_files)
        old = old[old.source == "clean"].groupby(TEAM)["return"].mean()
        both = pd.concat([j0, old.rename("old")], axis=1, join="inner")
        print(f"clean-return reproduction vs Phase 1 Stage 0: {len(both)} teams, "
              f"max |diff| = {(both.J0 - both.old).abs().max():.4f}")

    f = df[df.kind != "clean"].copy()
    f["direction"] = f.direction.fillna(-1)
    cell = ["kind", "strength", "scope", "direction"]
    m = f.groupby(TEAM + cell + ["predictor"]).agg(ret=("return", "mean"),
                                                   cshare=("clip_share", "mean")).reset_index()
    m = m.join(j0, on=TEAM)
    m["rho"] = (m.ret - m.task.map(J_RAND)) / (m.J0 - m.task.map(J_RAND))
    m.to_csv(os.path.join(a.dir, "per_team_cells.csv"), index=False)

    # collapse victims: all / worst1 / wdir, choosing the victim by "none"
    rows = []
    for key, g in m.groupby(TEAM + ["kind", "strength"]):
        kind = key[3]
        if kind == "wdir":
            fam = {"wdir": g}
        else:
            fam = {"all": g[g.scope == "all"], "worst1": g[g.scope != "all"]}
        for scope, h in fam.items():
            if h.empty:
                continue
            none = h[h.predictor == "none"]
            pick = none.loc[none.rho.idxmin(), ["scope", "direction"]]
            sel = h[(h.scope == pick.scope) & (h.direction == pick.direction)]
            for _, r in sel.iterrows():
                rows.append({**dict(zip(TEAM + ["kind", "strength"], key)), "scope": scope,
                             "victim": f"{pick.scope}/d{int(pick.direction)}",
                             "predictor": r.predictor, "rho": r.rho, "cshare": r.cshare})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.dir, "per_team.csv"), index=False)

    # rho under "none" for every method: the first row of the Phase 2 results
    none = t[t.predictor == "none"]
    out = []
    for key, g in none.groupby(["method", "kind", "strength", "scope"]):
        lo, hi = boot(g.rho.values)
        out.append(dict(zip(["method", "kind", "strength", "scope"], key), n=len(g),
                        rho=g.rho.mean(), lo=lo, hi=hi))
    rho = pd.DataFrame(out)
    rho.to_csv(os.path.join(a.dir, "rho_none.csv"), index=False)

    # oracle gains, paired per team
    w = t.pivot_table(index=TEAM + ["kind", "strength", "scope"], columns="predictor",
                      values="rho").reset_index()
    clip = t[t.predictor == "exact"].set_index(TEAM + ["kind", "strength", "scope"])["cshare"]
    gains = []
    for key, g in w.groupby(["method", "kind", "strength", "scope"]):
        r = dict(zip(["method", "kind", "strength", "scope"], key), n=len(g),
                 rho_none=g["none"].mean())
        for pred in ("exact", "d1", "d3", "d10"):
            if pred in g and g[pred].notna().all():
                d = g[pred] - g["none"]
                r[f"gain_{pred}"] = d.mean()
                r[f"lo_{pred}"], r[f"hi_{pred}"] = boot(d.values)
        idx = [tuple(x) for x in g[TEAM + ["kind", "strength", "scope"]].values]
        r["clip_exact"] = clip.reindex(idx).mean()
        gains.append(r)
    gains = pd.DataFrame(gains)
    gains.to_csv(os.path.join(a.dir, "oracle_gains.csv"), index=False)

    pd.set_option("display.width", 250)
    print("clean J0 by method:", j0.groupby("method").mean().round(2).to_dict())
    print("\nrho under no correction (mean over seeds):")
    print(rho.pivot_table(index=["kind", "strength", "scope"], columns="method",
                          values="rho").round(3).to_string())
    cols = ["method", "kind", "strength", "scope", "rho_none", "gain_exact", "lo_exact",
            "hi_exact", "gain_d1", "gain_d3", "gain_d10", "clip_exact"]
    print("\noracle gains:")
    print(gains[[c for c in cols if c in gains]].round(3).to_string(index=False))

    mad = gains[(gains.method == "maddpg") & gains.kind.isin(["gain", "bias"])]
    best = mad.loc[mad.gain_exact.idxmax()]
    go = (mad.gain_exact >= a.threshold).any()
    print(f"\nSTOP RULE: max MADDPG exact-oracle gain on gain/bias cells = {best.gain_exact:.3f} "
          f"({best.kind} {best.strength} {best.scope}; threshold {a.threshold}) "
          f"-> {'GO' if go else 'STOP (spread arm)'}")


if __name__ == "__main__":
    main()
