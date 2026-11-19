"""Phase 1 Stage 0: checks on a Phase 0 checkpoint, no training.

1. Saturation: share of intended actions with |a| <= threshold, per dimension, clean and under
   NR with random replacement at alpha 0.3 (all agents).
2. Adversary speed: per-step change ||r_t - r_{t-1}|| and lag-1 autocorrelation of the
   replacement action r, for the learned NR adversaries trained in Phase 0, i.i.d. random R and
   correlated random R_phi.
3. Oracle NR compensation: the frozen team's intended action a* is replaced by the command
       a = clip((a* - alpha * r_hat) / (1 - alpha), -1, 1)
   so that the executed action (1 - alpha) a + alpha r equals a* when r_hat = r.
   Predictors r_hat: none (no correction, the Phase 0 setting), zero (gain correction only),
   last (r_{t-1}, recoverable from the agent's own motion), exact (r_t, clairvoyant ceiling).

Sources of r:  R      r_t ~ U([-1,1]^2), i.i.d.
               Rphi   r_t = clip(phi r_{t-1} + sqrt(1 - phi^2) u_t), u_t ~ U([-1,1]^2)
               L      Phase 0 learned NR adversary r = nu(s), loaded from <run>/adv/
Common random numbers: the uniform draws depend only on (source, phi, alpha, episode), never on
the predictor, and evaluation seeds are Phase 0's (100_000 + episode).

Writes <out>/<task>__<method>__s<seed>.csv (one row per episode) and .json (saturation, speed).
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from marl.attack import load_team
from marl.envs import make_env
from marl.nets import Actor

PREDICTORS = ("none", "zero", "last", "exact")
SAT_THRESHOLDS = (0.9, 0.8, 0.7, 0.5, 0.4, 0.0)  # 1-alpha and 1-2alpha for alpha 0.1, 0.3, 0.5


def load_adversary(path, obs_dim, n_victims):
    adv = Actor(obs_dim, 2 * n_victims, 128)
    adv.load_state_dict(torch.load(path, map_location="cpu"))
    adv.eval()

    def act(obs):
        with torch.no_grad():
            x = torch.as_tensor(np.concatenate(obs), dtype=torch.float32).unsqueeze(0)
            return adv(x)[0].numpy().reshape(n_victims, 2)
    return act


def episode(env, team, n, alpha, victims, source, predictor, ep_seed, rng_key, phi=0.0,
            adv=None, record=None):
    """Return the team return of one episode; optionally record intended actions and r."""
    rng = np.random.default_rng(rng_key)
    obs = env.reset(seed=ep_seed)
    r_prev = np.zeros((n, 2), np.float32)
    total, done, t = 0.0, False, 0
    while not done:
        a_star = team(obs)
        u = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        if source == "R":
            r = u
        elif source == "Rphi":
            r = u if t == 0 else np.clip(phi * r_prev + np.sqrt(1 - phi ** 2) * u, -1, 1)
        elif source == "L":
            r = np.zeros((n, 2), np.float32)
            r[victims] = adv(obs)
        else:  # clean
            r = None
        cmd = a_star.copy()
        if r is not None and predictor != "none":
            r_hat = {"zero": np.zeros_like(r), "last": r_prev, "exact": r}[predictor]
            for j in victims:
                cmd[j] = np.clip((a_star[j] - alpha * r_hat[j]) / (1 - alpha), -1, 1)
        exe = cmd.copy()
        if r is not None:
            for j in victims:
                exe[j] = (1 - alpha) * cmd[j] + alpha * r[j]
            exe = np.clip(exe, -1, 1)
        if record is not None:
            record["a"].append(a_star.copy())
            if r is not None:
                record["r"].append(r[victims].copy())
        obs, _, team_r, done = env.step(exe)
        total += team_r
        if r is not None:
            r_prev = r
        t += 1
    return total


def speed_stats(seqs):
    """seqs: list of arrays (T, V, 2). Mean step change and lag-1 autocorrelation."""
    d = np.concatenate([np.linalg.norm(s[1:] - s[:-1], axis=-1).ravel() for s in seqs])
    x = np.concatenate([s[:-1].reshape(-1) for s in seqs])
    y = np.concatenate([s[1:].reshape(-1) for s in seqs])
    return {"mean_step_change": float(d.mean()), "lag1_autocorr": float(np.corrcoef(x, y)[0, 1]),
            "mean_abs_r": float(np.abs(np.concatenate([s.ravel() for s in seqs])).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--alphas", default="0.1,0.3,0.5")
    p.add_argument("--phis", default="0.9,0.99")
    p.add_argument("--adv-alphas", default="0.1,0.3")
    a = p.parse_args()
    torch.set_num_threads(1)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    env = make_env(cfg["task"], prey_path=cfg.get("prey", "assets/tag_prey.pt"))
    n = env.n_team
    team = load_team(a.run_dir, n)
    obs_dim = sum(env.obs_dims[:n])
    alphas = [float(x) for x in a.alphas.split(",")]
    phis = [float(x) for x in a.phis.split(",")]
    all_v = list(range(n))

    cells = [("clean", 0.0, 0.0, "all", all_v, None, -1)]
    for ai, al in enumerate(alphas):
        cells.append(("R", 0.0, al, "all", all_v, None, -1))
        for ph in phis:
            cells.append(("Rphi", ph, al, "all", all_v, None, -1))
    for al in [float(x) for x in a.adv_alphas.split(",")]:
        for scope, victims in (("all", all_v), ("one", [0])):
            for s in (0, 1):
                path = os.path.join(a.run_dir, "adv", f"nr_a{al}_{scope}_s{s}.pt")
                if os.path.exists(path):
                    cells.append(("L", 0.0, al, scope, victims,
                                  load_adversary(path, obs_dim, len(victims)), s))

    src_id = {"clean": 0, "R": 1, "Rphi": 2, "L": 3}
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"{cfg['task']}__{cfg['method']}__s{cfg['seed']}")
    sat, speed = {}, {}
    with open(stem + ".csv.tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "seed", "source", "phi", "alpha", "scope", "adv_seed",
                    "predictor", "episode", "return"])
        for source, phi, al, scope, victims, adv, adv_seed in cells:
            preds = ("none",) if source == "clean" else PREDICTORS
            key_cell = f"{source}_phi{phi}_a{al}_{scope}_adv{adv_seed}"
            rec_a, rec_r = [], []
            for pred in preds:
                for ep in range(a.episodes):
                    rec = {"a": [], "r": []} if pred == "none" else None
                    ret = episode(env, team, n, al, victims, source, pred, 100_000 + ep,
                                  [src_id[source], int(phi * 100), int(al * 100), ep],
                                  phi=phi, adv=adv, record=rec)
                    if rec is not None:
                        rec_a.append(np.array(rec["a"]))
                        if rec["r"]:
                            rec_r.append(np.array(rec["r"]))
                    w.writerow([cfg["task"], cfg["method"], cfg["seed"], source, phi, al, scope,
                                adv_seed, pred, ep, round(ret, 4)])
            if source == "clean" or (source == "R" and al == 0.3):
                acts = np.abs(np.concatenate(rec_a).reshape(-1))
                sat[key_cell] = {f"le_{th}": float((acts <= th).mean()) for th in SAT_THRESHOLDS}
            if rec_r:
                speed[key_cell] = speed_stats(rec_r)
    os.replace(stem + ".csv.tmp", stem + ".csv")
    json.dump({"task": cfg["task"], "method": cfg["method"], "seed": cfg["seed"],
               "saturation": sat, "speed": speed}, open(stem + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
