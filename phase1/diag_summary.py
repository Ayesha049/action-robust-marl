"""Summarise diag_stage1 outputs (see diag_stage1.py for definitions). Means over seeds 0-2."""

import argparse
import glob
import os

import numpy as np
import pandas as pd

J_RAND = -26.08


def fit(g):
    """delta ~ k_amp * mu_b + k_comp * (-r_prev), pooled over both components, t >= 1."""
    g = g[g.t >= 1]
    y = np.r_[g.cmd0 - g.mub0, g.cmd1 - g.mub1]
    X = np.c_[np.r_[g.mub0, g.mub1], -np.r_[g.rprev0, g.rprev1]]
    k, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ k) ** 2).sum() / (y ** 2).sum()
    Xe = np.c_[np.r_[g.mub0, g.mub1], -np.r_[g.r0, g.r1]]
    ke, *_ = np.linalg.lstsq(Xe, y, rcond=None)
    return pd.Series({"k_amp": k[0], "k_comp": k[1], "R2": r2, "k_comp_exact": ke[1]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="runs_phase1/diag")
    a = p.parse_args()
    df = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(a.dir, "diag_*.csv")))
    df["kind"] = df.label.str.rsplit("_s", n=1).str[0] + np.where(df.blind == 1, "-blind", "")
    df["seed"] = df.label.str.rsplit("_s", n=1).str[1].astype(int)
    df["cmd_n"] = np.hypot(df.cmd0, df.cmd1)
    df["mub_n"] = np.hypot(df.mub0, df.mub1)
    df["d_n"] = np.hypot(df.cmd0 - df.mub0, df.cmd1 - df.mub1)
    df["sat"] = ((df.cmd0.abs() > 0.95).astype(int) + (df.cmd1.abs() > 0.95)) / 2
    df["exe_sat"] = ((df.exe0.abs() > 0.95).astype(int) + (df.exe1.abs() > 0.95)) / 2
    key = ["kind", "seed", "cell"]
    step = df.groupby(key)[["cmd_n", "mub_n", "d_n", "sat", "exe_sat", "speed"]].mean()
    a0 = df[df.agent == 0]
    epi = a0.groupby(key + ["episode"]).agg(ret=("team_r", "sum"), coll=("coll", "sum"),
                                            cover_end=("cover", "last"))
    epi = epi.groupby(key).mean()
    j0 = epi.xs("clean", level="cell").ret.rename("J0")
    epi = epi.join(j0, on=["kind", "seed"])
    epi["rho"] = (epi.ret - J_RAND) / (epi.J0 - J_RAND)
    nr = df[df.cell != "PR-R"]
    reg = nr.groupby(key).apply(fit)
    tab = step.join(epi[["rho", "coll", "cover_end"]]).join(reg)
    tab.reset_index().to_csv(os.path.join(a.dir, "diag_per_seed.csv"), index=False)
    m = tab.groupby(["cell", "kind"]).mean()
    m.reset_index().to_csv(os.path.join(a.dir, "diag_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(m.round(3).to_string())
    prof = df[df.kind.isin(["residual", "residual-blind", "nr", "ft"])].groupby(
        ["kind", "cell", "t"]).d_n.mean().unstack("t")[[0, 1, 2, 3, 5, 10, 24]]
    prof.to_csv(os.path.join(a.dir, "diag_delta_profile.csv"))
    print("\n|delta| by time step\n", prof.round(3).to_string())
    bl = tab.rho.unstack("kind")
    d = (bl["residual"] - bl["residual-blind"]).groupby("cell").agg(["mean", "min", "max"])
    print("\nrho residual - residual-blind (mean, min, max over seeds)\n", d.round(3).to_string())


if __name__ == "__main__":
    main()
