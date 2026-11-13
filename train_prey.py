"""Fig. 3 analogue of Li et al. 2019 (M3DDPG): a DDPG prey trained from scratch
against a frozen predator team on simple_tag.

The prey's reward is the negative of the predators' reward (zero-sum), as in the
paper. The predator team is loaded from <run_dir>/actor_*.pt (trained in --mode all
self-play). Logs the predators' mean return over the last 100 training episodes every
25k steps, then evaluates the final deterministic prey over 100 episodes.

Writes <run_dir>/prey_s<seed>.json
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from marl.attack import load_team, run_episode
from marl.envs import make_env
from marl.nets import Actor, Critic, soft_update

PREY = "agent_0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(2000 + a.seed)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    assert cfg["task"] == "tag"

    env = make_env("tag", prey_path=cfg.get("prey", "assets/tag_prey.pt"))  # placeholder prey
    team = load_team(a.run_dir, env.n_team)
    hold = {"u": np.zeros(2, np.float32)}
    env.prey_policy = lambda o: hold["u"]  # prey action is injected before each step

    env.reset(seed=300_000 + a.seed)
    od = env._obs[PREY].shape[0]
    actor, critic = Actor(od, 2, 128), Critic(od + 2, 128)
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    c_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    S = a.steps
    O = np.zeros((S, od), np.float32)
    O2 = np.zeros_like(O)
    U = np.zeros((S, 2), np.float32)
    R = np.zeros((S, 1), np.float32)
    obs = env.reset(seed=300_000 + a.seed)
    ep_ret, rets, curve = 0.0, [], []
    for step in range(S):
        po = env._obs[PREY].astype(np.float32)
        if step < a.learning_starts:
            u = rng.uniform(-1, 1, 2).astype(np.float32)
        else:
            with torch.no_grad():
                u = actor(torch.as_tensor(po).unsqueeze(0))[0].numpy()
            u = np.clip(u + rng.normal(0, 0.2, 2), -1, 1).astype(np.float32)
        hold["u"] = u
        obs, _, team_r, done = env.step(team(obs))
        O[step], U[step], R[step], O2[step] = po, u, -team_r, env._obs[PREY]
        ep_ret += team_r
        if done:
            rets.append(ep_ret)
            ep_ret = 0.0
            obs = env.reset()
        if step >= a.learning_starts:
            idx = np.random.randint(0, step + 1, 256)
            o, u_, r, o2 = (torch.as_tensor(x[idx]) for x in (O, U, R, O2))
            with torch.no_grad():
                y = r + 0.95 * critic_t(o2, actor_t(o2))
            loss = F.mse_loss(critic(o, u_), y)
            c_opt.zero_grad()
            loss.backward()
            c_opt.step()
            if step % 2 == 0:
                pl = -critic(o, actor(o)).mean()
                a_opt.zero_grad()
                pl.backward()
                a_opt.step()
                soft_update(actor_t, actor, 0.005)
                soft_update(critic_t, critic, 0.005)
        if (step + 1) % 25_000 == 0:
            curve.append([step + 1, round(float(np.mean(rets[-100:])), 3)])

    actor.eval()

    def prey_det(o):
        with torch.no_grad():
            return actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))[0].numpy()
    env.prey_policy = prey_det
    final = [run_episode(env, team, None, seed=100_000 + ep, pert_seed=[0, 0, ep])
             for ep in range(a.eval_episodes)]
    json.dump({"task": "tag", "method": cfg["method"], "seed": cfg["seed"], "prey_seed": a.seed,
               "steps": a.steps, "final_mean_return": float(np.mean(final)), "curve": curve},
              open(os.path.join(a.run_dir, f"prey_s{a.seed}.json"), "w"))


if __name__ == "__main__":
    main()
