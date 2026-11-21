"""Train a team on an MPE task with one of: maddpg m3ddpg ernie pr nr marta mir3.

Writes runs/<task>__<method>__s<seed>/ with config.json, train_log.csv,
actor_<i>.pt for each team agent, and DONE.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import time

import numpy as np
import torch

from marl.envs import make_env
from marl.maddpg import METHODS, MADDPG, Replay
from marl.nets import save_actor

DEFAULTS = dict(hidden=64, lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01, batch=1024,
                buffer=1_000_000, noise=0.1, learning_starts=25_000, update_every=25,
                grad_steps=1, alpha=0.1, adv_ratio=None, m3_eps=0.01, ernie_lam=0.1,
                ernie_eps=0.1, ernie_steps=3, marta_cost=0.5, sw_eps=0.1, shared_reward=True,
                m3_actor=True, mir3_lam=0.01, mir3_window=3000, mir3_batch=64)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["spread", "tag", "transport"], default="spread")
    p.add_argument("--method", choices=METHODS, default="maddpg")
    p.add_argument("--mode", choices=["team", "all"], default="team",
                   help="'all' also controls the prey (only for pre-training the tag prey)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_500_000)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--prey", default="assets/tag_prey.pt")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=v)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v) if v is not None else int,
                           default=v)
    a = p.parse_args()
    if a.adv_ratio is None:
        a.adv_ratio = 10 if a.method == "pr" else 1
    if a.mode == "all":
        a.shared_reward = False
    return a


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    a = parse()
    torch.set_num_threads(1)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    name = a.run_name or f"{a.task}__{a.method}__s{a.seed}"
    out = os.path.join(a.out_dir, name)
    os.makedirs(out, exist_ok=True)
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    meta = dict(vars(a), git_sha=git_sha(), torch=torch.__version__)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    env = make_env(a.task, mode=a.mode, prey_path=a.prey)
    agent = MADDPG(env.obs_dims, a.method, cfg, rng)
    buf = Replay(a.buffer, env.obs_dims, env.n, env.act_dim)

    logf = open(os.path.join(out, "train_log.csv"), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["step", "episodes", "team_return_mean100", "fault_frac", "sps"])
    obs = env.reset(seed=a.seed)
    ep_ret, ep_faults, ep_len, rets, faults, episodes = 0.0, 0, 0, [], [], 0
    start = time.time()
    for step in range(a.steps):
        if step < a.learning_starts:
            act, k = rng.uniform(-1, 1, (env.n, 2)).astype(np.float32), 0
        else:
            act, k = agent.act(obs, explore=True)
        obs2, r, team_r, done = env.step(act)
        buf.add(obs, act, r, team_r, obs2, k)
        obs = obs2
        ep_ret += team_r
        ep_faults += int(k > 0) if a.method == "marta" else 0
        ep_len += 1
        if done:
            rets.append(ep_ret)
            faults.append(ep_faults / ep_len)
            episodes += 1
            ep_ret, ep_faults, ep_len = 0.0, 0, 0
            obs = env.reset()
        if step >= a.learning_starts and step % a.update_every == 0:
            for _ in range(a.grad_steps):
                agent.update(buf)
        if (step + 1) % 25_000 == 0:
            log.writerow([step + 1, episodes, round(float(np.mean(rets[-100:])), 3),
                          round(float(np.mean(faults[-100:])), 3),
                          round((step + 1) / (time.time() - start), 1)])
            logf.flush()

    team_n = env.n_team if a.mode == "team" else env.n
    for i in range(team_n):
        save_actor(agent.actors[i], os.path.join(out, f"actor_{i}.pt"))
    if a.mode == "all" and a.task == "tag":
        save_actor(agent.actors[env.n - 1], os.path.join(out, "prey.pt"))
    with open(os.path.join(out, "DONE"), "w") as f:
        f.write(f"{time.time() - start:.0f} seconds\n")
    logf.close()
    env.close()


if __name__ == "__main__":
    main()
