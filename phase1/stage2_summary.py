"""Summarise Phase 1 Stage 2 and apply tests T1-T6 (framework doc 7.6, PHASE-1.md Stage 2).

Competence retained rho = (J - J_rand) / (J0 - J_rand), J0 = the team's own clean return,
J_rand = -26.08. Worst victim = minimum over one0..one2 of the per-seed mean rho. Learned cells:
mean over the 2 adversary seeds (and the worse of the two, reported). Paired by seed, 95%
bootstrap CI over seeds; better / worse / unclear as in Phase 0.

T1 L-NR all 0.3: method better than PR-MADDPG and NR-MADDPG
T2 NR cells (22): better than PR-MADDPG in a majority, worse in none
T3 slope of (method - FT) over phi in {0, 0.9, 0.99}, NR alpha 0.3 scope all: CI > 0
T4 PR cells (23): lower CI of method - PR-MADDPG > -0.03
T5 clean return: lower CI of method - PR-MADDPG > -0.5
T6 attribution on NR cells, reported: RC-FT, REC-FT, RC-REC, RC-RESMLP (and RC-fast - RC)
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

J_RAND = -26.08
DELTA, CLEAN_MARGIN = 0.03, 0.5
NEW = ["residual", "rcfast", "ft", "rec", "resmlp"]
REF = ["maddpg", "pr", "nr"]


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def load(out, runs):
    ev = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(out, "eval", "*.csv")))
    ev["kind"] = ev.label.str.rsplit("_s", n=1).str[0]
    ev["seed"] = ev.label.str.rsplit("_s", n=1).str[1].astype(int)
    m = ev.groupby(["kind", "seed", "source", "op", "phi", "scope", "alpha"])["return"].mean()
    m = m.reset_index()
    j0 = m[m.source == "clean"].set_index(["kind", "seed"])["return"].rename("J0")
    m = m.join(j0, on=["kind", "seed"])
    m["rho"] = (m["return"] - J_RAND) / (m.J0 - J_RAND)
    src = m.source + m.phi.map(lambda x: f"{x:g}" if x else "")
    m = m[m.source != "clean"].assign(src=src)
    one = m[m.scope.str.startswith("one")]
    worst = one.groupby(["kind", "seed", "src", "op", "alpha"]).rho.min().reset_index()
    worst["scope"] = "worst1"
    rnd = pd.concat([m[m.scope == "all"][worst.columns], worst])
    rnd["cell"] = rnd.src + "-" + rnd.op + "-" + rnd.scope + "-a" + rnd.alpha.map(lambda x: f"{x:g}")
    rows = [rnd[["kind", "seed", "cell", "rho"]].assign(rho_worse=np.nan)]
    adv = []
    for kind in NEW + REF:
        pat = (f"{out}/{kind}_s*/adv/*.json" if kind in NEW
               else f"{runs}/spread__{kind}__s*/adv/*.json")
        for f in glob.glob(pat):
            d = json.load(open(f))
            if d["alpha"] not in (0.1, 0.3, 1.0):
                continue
            seed = int(f.split("/")[-3].rsplit("s", 1)[1])
            adv.append({"kind": kind, "seed": seed, "advseed": d["adv_seed"],
                        "cell": f"L-{d['op']}-{d['scope']}-a{d['alpha']:g}", "ret": d["mean_return"]})
    if adv:
        adv = pd.DataFrame(adv).join(j0, on=["kind", "seed"])
        adv["rho"] = (adv.ret - J_RAND) / (adv.J0 - J_RAND)
        g = adv.groupby(["kind", "seed", "cell"]).rho
        rows.append(pd.concat([g.mean(), g.min().rename("rho_worse"), g.size().rename("n_adv")],
                              axis=1).reset_index())
    r = pd.concat(rows, ignore_index=True)
    r = r[r.seed.isin(j0.reset_index().seed.unique())]
    return r, j0.reset_index()


def paired(w, a, b):
    if a not in w or b not in w:
        return None
    d = (w[a] - w[b]).dropna()
    if len(d) < 2:
        return None
    lo, hi = boot(d.values)
    v = "better" if lo > 0 else "worse" if hi < 0 else "unclear"
    return dict(mean=d.mean(), lo=lo, hi=hi, n=len(d), verdict=v)


def tests(r, j0, m):
    nr_cells = sorted(c for c in r.cell.unique() if "-nr-" in c and not c.startswith("L-")) + \
        [f"L-nr-{s}-a{a:g}" for s in ("all", "one") for a in (0.1, 0.3)]
    pr_cells = sorted(c for c in r.cell.unique() if "-pr-" in c and not c.startswith("L-")) + \
        [f"L-pr-{s}-a{a:g}" for s in ("all", "one") for a in (0.1, 0.3)] + ["L-pr-one-a1"]
    w = r.pivot_table(index=["cell", "seed"], columns="kind", values="rho")
    cellres = lambda c, a, b: paired(w.xs(c, level="cell"), a, b) if c in w.index.levels[0] else None
    out = {}
    t1a, t1b = cellres("L-nr-all-a0.3", m, "pr"), cellres("L-nr-all-a0.3", m, "nr")
    out["T1"] = bool(t1a and t1b and t1a["verdict"] == "better" and t1b["verdict"] == "better")
    v2 = [cellres(c, m, "pr") for c in nr_cells]
    nb = sum(1 for x in v2 if x and x["verdict"] == "better")
    nw = sum(1 for x in v2 if x and x["verdict"] == "worse")
    nmiss = sum(1 for x in v2 if x is None)
    out["T2"] = nb > len(nr_cells) / 2 and nw == 0 and nmiss == 0
    # T3: per-seed slope of (m - ft) over phi
    phis = np.array([0.0, 0.9, 0.99])
    cs = ["R-nr-all-a0.3", "Rphi0.9-nr-all-a0.3", "Rphi0.99-nr-all-a0.3"]
    sl = None
    if "ft" in w and m in w:
        d = pd.concat([(w.xs(c, level="cell")[m] - w.xs(c, level="cell")["ft"]).rename(c)
                       for c in cs], axis=1).dropna()
        sl = np.array([np.polyfit(phis, row.values, 1)[0] for _, row in d.iterrows()])
        lo, hi = boot(sl)
        out["T3"] = lo > 0
        out["T3_detail"] = f"slope {sl.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]; diffs " + \
            ", ".join(f"{d[c].mean():+.4f}" for c in cs)
    else:
        out["T3"] = False
    v4 = [(c, cellres(c, m, "pr")) for c in pr_cells]
    fails4 = [c for c, x in v4 if x is None or x["lo"] <= -DELTA]
    out["T4"] = not fails4
    cl = j0.pivot_table(index="seed", columns="kind", values="J0")
    t5 = paired(cl, m, "pr")
    out["T5"] = bool(t5 and t5["lo"] > -CLEAN_MARGIN)
    label = ("Full" if all(out[k] for k in ("T1", "T2", "T3", "T4", "T5")) else
             "Partial" if all(out[k] for k in ("T2", "T3", "T4", "T5")) else
             "Mechanism only" if out["T3"] and out["T4"] and out["T5"] else "No")
    lines = [f"== {m}: outcome {label}",
             f"T1 L-NR all 0.3: vs PR {fmt(t1a)}; vs NR {fmt(t1b)} -> {ok(out['T1'])}",
             f"T2 NR cells: better {nb}, worse {nw}, missing {nmiss} of {len(nr_cells)} -> {ok(out['T2'])}",
             f"T3 {out.get('T3_detail', 'missing')} -> {ok(out['T3'])}",
             f"T4 PR cells: {len(pr_cells) - len(fails4)} of {len(pr_cells)} pass; fail: {fails4} -> {ok(out['T4'])}",
             f"T5 clean return vs PR: {fmt(t5)} -> {ok(out['T5'])}"]
    return out, label, lines


def fmt(x):
    return "missing" if x is None else f"{x['mean']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] {x['verdict']}"


def ok(b):
    return "pass" if b else "fail"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs_phase1s2")
    p.add_argument("--runs", default="runs")
    a = p.parse_args()
    r, j0 = load(a.out, a.runs)
    r.to_csv(os.path.join(a.out, "stage2_per_team.csv"), index=False)
    w = r.pivot_table(index=["cell", "seed"], columns="kind", values="rho")
    rows = []
    for cell, g in w.groupby(level=0):
        row = {"cell": cell}
        for k in REF + NEW:
            if k in g:
                row[k] = g[k].mean()
        for x, y in [("residual", "pr"), ("residual", "nr"), ("rcfast", "pr"), ("rcfast", "residual"),
                     ("residual", "ft"), ("rec", "ft"), ("residual", "rec"), ("residual", "resmlp")]:
            res = paired(g.droplevel(0), x, y)
            if res:
                row[f"{x}-{y}"] = res["mean"]
                row[f"{x}-{y}_ci"] = f"[{res['lo']:+.3f}, {res['hi']:+.3f}] {res['verdict']}"
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.out, "stage2_summary.csv"), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 200)
    cols = ["cell"] + [k for k in REF + NEW if k in s]
    print(s[cols].round(3).to_string(index=False))
    print("\nclean return:", j0.groupby("kind").J0.mean().round(3).to_dict())
    lines = []
    for m in ("residual", "rcfast", "rec", "resmlp", "ft"):
        if m in w:
            _, _, ln = tests(r, j0, m)
            lines += ln + [""]
    # T6: attribution averaged over the NR cells, per seed
    nr = [c for c in w.index.levels[0] if "-nr-" in c]
    wn = w.loc[nr].groupby(level="seed").mean()
    lines.append("T6 attribution, mean over NR cells (paired by seed):")
    for x, y in [("residual", "ft"), ("rec", "ft"), ("residual", "rec"), ("residual", "resmlp"),
                 ("rcfast", "residual"), ("rcfast", "ft")]:
        lines.append(f"  {x} - {y}: {fmt(paired(wn, x, y))}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(a.out, "stage2_tests.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
