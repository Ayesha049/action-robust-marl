"""Pick each baseline's hyperparameters from the tuning runs by CLEAN return.

Each tuning run is evaluated with no attack (50 episodes, fixed seeds); the
config with the best mean clean return across tuning seeds wins. Evaluation
attacks are never used for selection. Writes:
  <out>/hparams_<method>.txt   extra train.py flags for that method
  <out>/tuning_<method>.csv    every config's clean return (sensitivity table)
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from marl.attack import load_team, run_episode
from marl.envs import make_env

TUNED = {"m3ddpg": ["m3_eps"], "ernie": ["ernie_lam", "ernie_eps"], "marta": ["marta_cost"],
         "mir3": ["mir3_lam"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tune-root", default="runs_tune")
    p.add_argument("--method", required=True)
    p.add_argument("--out", default="runs")
    p.add_argument("--episodes", type=int, default=50)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    flag_file = os.path.join(a.out, f"hparams_{a.method}.txt")
    if a.method not in TUNED:
        open(flag_file, "w").write("")
        return
    env = make_env("spread")
    rows = []
    for d in sorted(glob.glob(os.path.join(a.tune_root, "*", f"spread__{a.method}__s*"))):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        team = load_team(d, env.n_team)
        ret = np.mean([run_episode(env, team, None, seed=300_000 + e, pert_seed=0)
                       for e in range(a.episodes)])
        rows.append({**{k: cfg[k] for k in TUNED[a.method]}, "seed": cfg["seed"], "clean": ret})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"tuning_{a.method}.csv"), index=False)
    best = df.groupby(TUNED[a.method])["clean"].mean().idxmax()
    best = best if isinstance(best, tuple) else (best,)
    flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in zip(TUNED[a.method], best))
    open(flag_file, "w").write(flags)
    print(a.method, "->", flags)


if __name__ == "__main__":
    main()
