"""Learned (L) attacker against a frozen team that may be recurrent (Phase 1, Stage 1).

Same algorithm, budgets and evaluation as Phase 0's train_adversary.py (300k steps, DDPG,
reward = -team reward, 100 evaluation episodes, seeds 100_000 + episode). The only change: the
team is loaded with phase1.residual.load_team_any and is told its executed action after every
step (team.observe), which a residual policy needs as evidence; Phase 0 teams ignore it.
The original Phase 0 description follows.

Learned (L) attacker: a DDPG adversary trained against a frozen team.

The adversary sees the global observation and outputs replacement actions for
the victims; the PR/NR operator with budget alpha decides what is executed.
Reward = -team reward. It's never seen by any method during training. After
training it's evaluated over 100 episodes with common random numbers.

Writes <run_dir>/adv/<op>_a<alpha>_<scope>_s<seed>.json
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from marl.attack import Attack
from phase1.residual import load_team_any
from marl.envs import make_env
from marl.nets import Actor, Critic, soft_update

OPS = {"pr": 0, "nr": 1}


def run_episode_team(env, team, attack, seed, pert_seed):
    """marl.attack.run_episode for a team with reset/act/observe."""
    rng = np.random.default_rng(pert_seed)
    obs = env.reset(seed=seed)
    team.reset()
    total, done = 0.0, False
    while not done:
        a = team.act(obs)
        exe = attack.apply(obs, a, attack.draw(rng))
        team.observe(a, exe.astype(np.float32))
        obs, _, team_r, done = env.step(exe)
        total += team_r
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--op", choices=["pr", "nr"], required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--scope", choices=["one", "all"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(1000 + a.seed)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    cfg.setdefault("method", cfg.get("kind"))
    env = make_env(cfg["task"], prey_path=cfg.get("prey", "assets/tag_prey.pt"))
    n = env.n_team
    team = load_team_any(a.run_dir, n)
    victims = [0] if a.scope == "one" else list(range(n))
    obs_dim, act_dim = sum(env.obs_dims[:n]), 2 * len(victims)

    actor = Actor(obs_dim, act_dim, 128)
    critic = Critic(obs_dim + act_dim, 128)
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    c_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    att = Attack(a.op, a.alpha, victims, "R", n)  # used only for _execute()

    size = a.steps
    O = np.zeros((size, obs_dim), np.float32); O2 = np.zeros_like(O)
    U = np.zeros((size, act_dim), np.float32); R = np.zeros((size, 1), np.float32)
    obs = env.reset(seed=200_000 + a.seed)
    team.reset()
    for step in range(a.steps):
        oc = np.concatenate(obs)
        if step < 10_000:
            u = rng.uniform(-1, 1, act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                u = actor(torch.as_tensor(oc).unsqueeze(0))[0].numpy()
            u = np.clip(u + rng.normal(0, 0.2, act_dim), -1, 1).astype(np.float32)
        intended = team.act(obs)
        repl = intended.copy()
        repl[victims] = u.reshape(len(victims), 2)
        coins = rng.random(n)
        exe = att._execute(intended, repl, coins)
        team.observe(intended, exe.astype(np.float32))
        obs2, _, team_r, done = env.step(exe)
        O[step], U[step], R[step], O2[step] = oc, u, -team_r, np.concatenate(obs2)
        if done:
            obs = env.reset()
            team.reset()
        else:
            obs = obs2
        if step >= 10_000:
            idx = np.random.randint(0, step + 1, 256)
            o, u_, r, o2 = (torch.as_tensor(x[idx]) for x in (O, U, R, O2))
            with torch.no_grad():
                y = r + 0.95 * critic_t(o2, actor_t(o2))
            loss = F.mse_loss(critic(o, u_), y)
            c_opt.zero_grad(); loss.backward(); c_opt.step()
            if step % 2 == 0:
                pl = -critic(o, actor(o)).mean()
                a_opt.zero_grad(); pl.backward(); a_opt.step()
                soft_update(actor_t, actor, 0.005)
                soft_update(critic_t, critic, 0.005)

    actor.eval()
    ev = Attack(a.op, a.alpha, victims, "L", n, adversary=actor)
    rets = [run_episode_team(env, team, ev, seed=100_000 + ep, pert_seed=[OPS[a.op], 99, ep])
            for ep in range(a.eval_episodes)]
    os.makedirs(os.path.join(a.run_dir, "adv"), exist_ok=True)
    name = f"{a.op}_a{a.alpha}_{a.scope}_s{a.seed}"
    torch.save(actor.state_dict(), os.path.join(a.run_dir, "adv", name + ".pt"))
    json.dump({"task": cfg["task"], "method": cfg["method"], "seed": cfg["seed"], "op": a.op,
               "alpha": a.alpha, "scope": a.scope, "adv_seed": a.seed,
               "mean_return": float(np.mean(rets)), "returns": [round(x, 4) for x in rets]},
              open(os.path.join(a.run_dir, "adv", name + ".json"), "w"))


if __name__ == "__main__":
    main()
