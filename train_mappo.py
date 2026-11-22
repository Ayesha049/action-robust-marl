"""Train a MAPPO team on a VMAS task (Phase 2 Transport base). See marl/mappo.py.

Writes <out-dir>/<task>__mappo__s<seed>/ with config.json, train_log.csv, actor_<i>.pt (the
deterministic mean policy, MADDPG Actor format), mappo_full.pt (actors, log_std, critic) and DONE.

train_log.csv per iteration: step (frames per env-agent step, i.e. env steps over all envs),
episodes, team_return (mean episode return over the batch), delivered (share of envs with the
package on the goal at the last step; transport only), std (mean policy std), pi_loss, v_loss,
ratio_dev, sps.
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

from marl.envs import TRANSPORT_STEPS
from marl.mappo import MAPPO
from marl.nets import save_actor

DEFAULTS = dict(hidden=256, lr=5e-5, clip=0.2, epochs=45, minibatch=4096, gamma=0.99, lam=0.9,
                ent_coef=0.0, vf_coef=1.0, max_grad=1.0, init_log_std=0.0, n_envs=600,
                horizon=TRANSPORT_STEPS, n_agents=4, threads=4)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["transport", "balance"], default="transport")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=24_000_000, help="total env frames (all envs)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--no-share", action="store_true")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    import vmas
    env = vmas.make_env(a.task, num_envs=a.n_envs, device="cpu", continuous_actions=True,
                        seed=a.seed, n_agents=a.n_agents)
    n = a.n_agents
    obs_dim = int(env.observation_space[0].shape[0])
    algo = MAPPO(n, obs_dim, 2, a.hidden, a.lr, a.clip, a.epochs, a.minibatch, a.gamma, a.lam,
                 a.ent_coef, a.vf_coef, a.max_grad, a.init_log_std, share=not a.no_share)

    out = os.path.join(a.out_dir, f"{a.task}__mappo__s{a.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(vars(a), method="mappo", git_sha=git_sha(), torch=torch.__version__), f, indent=2)
    logf = open(os.path.join(out, "train_log.csv"), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["step", "episodes", "team_return", "delivered", "std", "pi_loss", "v_loss",
                  "ratio_dev", "sps"])

    def save():
        for i in range(n):
            save_actor(algo.actor(i), os.path.join(out, f"actor_{i}.pt"))
        torch.save({"actors": [x.state_dict() for x in algo.actors], "log_std": algo.log_std.data,
                    "critic": algo.critic.state_dict(), "share": algo.share}, os.path.join(out, "mappo_full.pt"))

    T, E = a.horizon, a.n_envs
    frames, episodes, it, t_start = 0, 0, 0, time.time()
    while frames < a.steps:
        t0 = time.time()
        o = torch.stack(env.reset(), 1)  # [E, n, d]
        O, U, LP, R = [], [], [], []
        for _ in range(T):
            u, act, lp = algo.act(o)
            nxt, rews, _, _ = env.step([act[:, i] for i in range(n)])
            O.append(o), U.append(u), LP.append(lp), R.append(torch.stack(rews, 1).mean(1))
            o = torch.stack(nxt, 1)
        O, U, LP, R = torch.stack(O), torch.stack(U), torch.stack(LP), torch.stack(R)
        stats = algo.update(O, U, LP, R, o)
        frames += T * E
        episodes += E
        it += 1
        delivered = (o[:, 0, 10] > 0.5).float().mean().item() if a.task == "transport" else float("nan")
        log.writerow([frames, episodes, round(R.sum(0).mean().item(), 4), round(delivered, 4),
                      round(algo.std().mean().item(), 4), *[round(float(x), 5) for x in stats],
                      round(T * E / (time.time() - t0), 1)])
        logf.flush()
        if it % 10 == 0:
            save()
    save()
    open(os.path.join(out, "DONE"), "w").close()
    print(f"done {frames} frames in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
