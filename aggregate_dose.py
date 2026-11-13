"""Dose-response on spread: robustness vs regularisation strength for M3DDPG and ERNIE-A-c.

Metric definitions are identical to aggregate.py (competence retained against the random
team, AUC over alpha 0..0.5). Reference points: MADDPG and PR-MADDPG, seeds 0-4 from
Phase 0b. The Phase 0b winners (m3 eps 0.1; ernie lam 1.0 eps 0.05) are the "main" runs.

Writes <out>/dose_runs.csv, dose_summary.csv (mean over seeds; paired difference to
MADDPG with 95% bootstrap CI) and fig_dose.png.
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aggregate import boot_ci, random_team_return  # noqa: E402

CELLS = {  # name -> (source, op, scope family)
    "R_PR_all": ("R", "pr", "all"), "R_NR_all": ("R", "nr", "all"),
    "G_PR_all": ("G", "pr", "all"), "G_NR_all": ("G", "nr", "all"),
    "G_PR_one": ("G", "pr", "one"), "G_PR_worst1": ("G", "pr", "worst1"),
}
LEARNED = {"L_PR_a0.3_all": ("pr", 0.3, "all"), "L_NR_a0.3_all": ("nr", 0.3, "all"),
           "L_PR_a1.0_one": ("pr", 1.0, "one")}


def run_metrics(d, rand):
    cfg = json.load(open(os.path.join(d, "config.json")))
    df = pd.read_csv(os.path.join(d, "eval_curves.csv"))
    cur = df.groupby(["source", "op", "scope", "alpha"])["return"].mean().reset_index()
    clean = cur[cur.alpha == 0]["return"].mean()
    cur["ret"] = (cur["return"] - rand) / (clean - rand)
    out = dict(seed=cfg["seed"], clean=clean)
    for name, (src, op, fam) in CELLS.items():
        g = cur[(cur.source == src) & (cur.op == op)]
        if fam == "all":
            v = g[g.scope == "all"].sort_values("alpha")
        else:
            one = g[g.scope.str.startswith("one")].groupby("alpha")[["return", "ret"]]
            v = (one.mean() if fam == "one" else one.min()).reset_index()
        out[name] = v["ret"].mean()
        out[name + "_raw"] = v["return"].mean()
    for name, (op, al, sc) in LEARNED.items():
        js = [json.load(open(f)) for f in glob.glob(os.path.join(d, "adv", f"{op}_a{al}_{sc}_s*.json"))]
        if js:
            r = float(np.mean([j["mean_return"] for j in js]))
            out[name + "_raw"] = r
            out[name] = (r - rand) / (clean - rand)
    return out


def collect(label, family, strength, pattern, rand, rows):
    for d in sorted(glob.glob(pattern)):
        if os.path.exists(os.path.join(d, "eval_curves.csv")):
            rows.append(dict(label=label, family=family, strength=strength, **run_metrics(d, rand)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maddpg", required=True, help="glob of MADDPG spread run dirs")
    p.add_argument("--pr", required=True, help="glob of PR-MADDPG spread run dirs")
    p.add_argument("--m3-main", required=True, help="glob of the Phase 0b M3DDPG runs (eps 0.1)")
    p.add_argument("--ernie-main", required=True, help="glob of the Phase 0b ERNIE runs (lam 1, eps 0.05)")
    p.add_argument("--dose-roots", nargs="+", required=True, help="dirs holding m3_eps*/ and ernie_*/")
    p.add_argument("--out", default="results/dose")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rand = random_team_return("spread")

    rows = []
    collect("MADDPG", "ref", np.nan, a.maddpg, rand, rows)
    collect("PR-MADDPG", "ref", np.nan, a.pr, rand, rows)
    collect("m3 eps=0.1 (Ph.0b)", "m3ddpg", 0.1, a.m3_main, rand, rows)
    collect("ernie lam=1 eps=0.05 (Ph.0b)", "ernie", 1.0 * 0.05 ** 2, a.ernie_main, rand, rows)
    for root in a.dose_roots:
        for cdir in sorted(glob.glob(os.path.join(root, "*"))):
            name = os.path.basename(cdir)
            m = re.fullmatch(r"m3_eps([\d.]+)", name)
            e = re.fullmatch(r"ernie_l([\d.]+)_e([\d.]+)", name)
            if m:
                collect(f"m3 eps={m[1]}", "m3ddpg", float(m[1]), f"{cdir}/spread__*__s*", rand, rows)
            elif e:  # effective strength of the squared-difference regulariser ~ lam * eps^2
                collect(f"ernie lam={e[1]} eps={e[2]}", "ernie", float(e[1]) * float(e[2]) ** 2,
                        f"{cdir}/spread__*__s*", rand, rows)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(a.out, "dose_runs.csv"), index=False)

    metrics = ["clean"] + list(CELLS) + list(LEARNED)
    raw = [m + "_raw" for m in metrics[1:] if m + "_raw" in runs]
    base = runs[runs.label == "MADDPG"].set_index("seed")
    summ = []
    for (lab, fam, st), g in runs.groupby(["label", "family", "strength"], dropna=False):
        row = dict(label=lab, family=fam, strength=st, n=len(g))
        g = g.set_index("seed")
        for m in metrics + raw:
            if m not in g:
                continue
            row[m] = g[m].mean()
            if lab != "MADDPG":
                s = g.index.intersection(base.index)
                d = (g.loc[s, m] - base.loc[s, m]).dropna().values
                if len(d) >= 2:
                    lo, hi = boot_ci(d)
                    row[m + "_diff"], row[m + "_lo"], row[m + "_hi"] = d.mean(), lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ).sort_values(["family", "strength"])
    summ.to_csv(os.path.join(a.out, "dose_summary.csv"), index=False)

    # figure: metric vs strength, one row per family
    show = [("clean", "clean return"), ("G_PR_all", "G, PR, all agents (retained)"),
            ("L_PR_a0.3_all", "L, PR α0.3, all (retained)"),
            ("L_PR_a1.0_one", "L, PR takeover, one agent (retained)")]
    fig, axes = plt.subplots(2, len(show), figsize=(15, 6.2), sharex="row")
    refs = summ[summ.family == "ref"].set_index("label")
    for r, (fam, xl) in enumerate([("m3ddpg", "M3DDPG ε"), ("ernie", "ERNIE-A-c strength λ·ε²")]):
        s = summ[summ.family == fam].dropna(subset=["strength"]).sort_values("strength")
        for c, (m, title) in enumerate(show):
            ax = axes[r, c]
            if m in s:
                gg = runs[(runs.family == fam)]
                for st, g in gg.groupby("strength"):
                    ax.scatter([st] * len(g), g[m], s=10, color="#8c6bb1" if fam == "m3ddpg" else "#2ca02c",
                               alpha=0.35)
                ax.plot(s.strength, s[m], "-o", color="#8c6bb1" if fam == "m3ddpg" else "#2ca02c")
            for lab, col, ls in [("MADDPG", "#4c4c4c", "-"), ("PR-MADDPG", "#1f77b4", "--")]:
                if lab in refs.index and m in refs.columns:
                    ax.axhline(refs.loc[lab, m], color=col, ls=ls, lw=1.2, label=lab)
            ax.set_xscale("log")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=8)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle("spread: robustness vs regularisation strength (dots = seeds, line = mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_dose.png"), dpi=140)

    pd.set_option("display.width", 250)
    cols = ["label", "n", "clean", "G_PR_all", "G_NR_all", "L_PR_a0.3_all", "L_NR_a0.3_all", "L_PR_a1.0_one"]
    print(summ[[c for c in cols if c in summ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
