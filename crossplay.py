"""Fig. 2 analogue of Li et al. 2019 (M3DDPG): cross-play on simple_tag.

Populations are trained in full self-play (train.py --task tag --mode all): one
MADDPG population and one population per M3DDPG variant. For each variant V and seed s
the four matchups are scored by the predators' mean team return (the paper's "normal
agents"):

    MA vs MA, MA vs V, V vs MA, V vs V      (predators vs prey)

The paper's claim: V-vs-MA is the highest and MA-vs-V the lowest of the four.
Layout: <root>/<population>/tag__<method>__s<seed>/ with actor_0..2 and prey.pt.

Writes <out>/crossplay.csv (one row per variant, seed, matchup) and
<out>/crossplay_summary.csv (paired differences with 95% bootstrap CIs).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from marl.attack import load_team, run_episode
from marl.envs import make_env


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def find(root, pop, seed):
    hits = glob.glob(os.path.join(root, pop, f"tag__*__s{seed}"))
    assert len(hits) == 1, (root, pop, seed, hits)
    return hits[0]


def score(pred_dir, prey_dir, episodes):
    env = make_env("tag", prey_path=os.path.join(prey_dir, "prey.pt"))
    team = load_team(pred_dir, env.n_team)
    r = [run_episode(env, team, None, seed=100_000 + ep, pert_seed=[0, 0, ep])
         for ep in range(episodes)]
    env.close()
    return float(np.mean(r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs_val")
    p.add_argument("--baseline", default="ma", help="MADDPG population name")
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--out", default="results/val")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for v in a.variants:
        for s in a.seeds:
            ma, vv = find(a.root, a.baseline, s), find(a.root, v, s)
            for name, (pred, prey) in {"MA_vs_MA": (ma, ma), "MA_vs_V": (ma, vv),
                                       "V_vs_MA": (vv, ma), "V_vs_V": (vv, vv)}.items():
                rows.append(dict(variant=v, seed=s, matchup=name,
                                 predator_return=score(pred, prey, a.episodes)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "crossplay.csv"), index=False)

    w = df.pivot_table(index=["variant", "seed"], columns="matchup", values="predator_return")
    summ = []
    for v, g in w.groupby(level=0):
        row = dict(variant=v, n=len(g))
        for c in w.columns:
            row[c] = g[c].mean()
        # paper's claim: V_vs_MA highest, MA_vs_V lowest of the four
        row["paper_order_holds_seeds"] = int(((g.V_vs_MA >= g[["MA_vs_MA", "MA_vs_V", "V_vs_V"]].max(axis=1))
                                              & (g.MA_vs_V <= g[["MA_vs_MA", "V_vs_MA", "V_vs_V"]].min(axis=1))).sum())
        for name, d in {"pred_gain_V_vs_MA_minus_MA_vs_MA": g.V_vs_MA - g.MA_vs_MA,
                        "prey_gain_MA_vs_V_minus_MA_vs_MA": g.MA_vs_V - g.MA_vs_MA}.items():
            lo, hi = boot_ci(d.values)
            row[name], row[name + "_lo"], row[name + "_hi"] = d.mean(), lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.out, "crossplay_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
