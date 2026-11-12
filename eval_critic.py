"""Train a held-out evaluation critic Q_eval(s, a) for a frozen team.

Used only by the gradient worst-case attacker (source G). It is fitted the same
way for every method, so attack strength doesn't depend on each method's own
critic. Data: frozen-team rollouts where random agents' actions are perturbed
(PR or NR, alpha ~ U(0, 0.5), uniform random replacements), so the critic sees
off-policy actions. Target: r + gamma * Q_eval(s', pi(s')) (policy evaluation of
the nominal team).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from marl.attack import Attack, load_team
from marl.envs import make_env
from marl.nets import Critic, soft_update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-steps", type=int, default=200_000)
    p.add_argument("--grad-steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
    env = make_env(cfg["task"], prey_path=cfg.get("prey", "assets/tag_prey.pt"))
    n = env.n_team
    team = load_team(a.run_dir, n)

    O, A, R, O2, PI2 = [], [], [], [], []
    obs, steps = env.reset(seed=50_000), 0
    attack = None
    while steps < a.data_steps:
        if env.t == 0:
            k = rng.integers(1, n + 1)
            victims = rng.choice(n, k, replace=False)
            attack = Attack(rng.choice(["pr", "nr"]), float(rng.uniform(0, 0.5)), victims, "R", n)
        act = attack.apply(obs, team(obs), attack.draw(rng))
        obs2, _, team_r, done = env.step(act)
        O.append(np.concatenate(obs)); A.append(act.flatten()); R.append(team_r)
        O2.append(np.concatenate(obs2)); PI2.append(team(obs2).flatten())
        obs = env.reset(seed=50_000 + steps) if done else obs2
        steps += 1
    t = lambda x: torch.as_tensor(np.array(x, np.float32))
    O, A, R, O2, PI2 = t(O), t(A), t(R).unsqueeze(1), t(O2), t(PI2)

    q = Critic(O.shape[1] + A.shape[1], hidden=128)
    q_t = Critic(O.shape[1] + A.shape[1], hidden=128)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    gamma = cfg.get("gamma", 0.95)
    losses = []
    for g in range(a.grad_steps):
        idx = torch.randint(0, len(O), (1024,))
        with torch.no_grad():
            y = R[idx] + gamma * q_t(O2[idx], PI2[idx])
        loss = F.mse_loss(q(O[idx], A[idx]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        soft_update(q_t, q, 0.01)
        if g % 1000 == 0:
            losses.append(round(loss.item(), 5))
    torch.save({"in_dim": O.shape[1] + A.shape[1], "hidden": 128, "state": q.state_dict()},
               os.path.join(a.run_dir, "eval_critic.pt"))
    json.dump({"loss_every_1k": losses}, open(os.path.join(a.run_dir, "eval_critic_log.json"), "w"))


def load_eval_critic(run_dir):
    d = torch.load(os.path.join(run_dir, "eval_critic.pt"), map_location="cpu")
    q = Critic(d["in_dim"], d["hidden"])
    q.load_state_dict(d["state"])
    q.eval()
    for p_ in q.parameters():
        p_.requires_grad_(False)
    return q


if __name__ == "__main__":
    main()
