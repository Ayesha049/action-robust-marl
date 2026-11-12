"""Action-perturbation attacks for evaluation (PR / NR operators).

An attack cell = (op, alpha, victims, source):
  op      : "pr"  executed_j = replacement_j with prob. alpha, else intended_j
            "nr"  executed_j = (1-alpha) intended_j + alpha replacement_j
  victims : list of team-agent indices that can be perturbed
  source  : "R" uniform random in [-1,1]^2
            "G" white-box worst case: PGD minimising a held-out evaluation
                critic Q_eval(s, executed joint action) over the replacement
            "L" learned adversary policy (global obs -> victims' replacements)

Randomness: draw(rng) consumes the same numbers every step regardless of op,
source or policy, so all methods face identical coin flips / random actions.
"""

import numpy as np
import torch


class Attack:
    def __init__(self, op, alpha, victims, source, n, eval_critic=None, adversary=None,
                 pgd_steps=10, pgd_lr=0.25):
        assert op in ("pr", "nr") and source in ("R", "G", "L")
        self.op, self.alpha, self.victims, self.source = op, alpha, list(victims), source
        self.n = n
        self.q = eval_critic
        self.adv = adversary
        self.pgd_steps, self.pgd_lr = pgd_steps, pgd_lr

    def draw(self, rng):
        return rng.random(self.n), rng.uniform(-1, 1, (self.n, 2)).astype(np.float32)

    def _execute(self, intended, repl, coins):
        out = intended.copy()
        for j in self.victims:
            if self.op == "pr":
                if coins[j] < self.alpha:
                    out[j] = repl[j]
            else:
                out[j] = (1 - self.alpha) * intended[j] + self.alpha * repl[j]
        return np.clip(out, -1, 1)

    def _worst_case(self, obs, intended, coins):
        active = [j for j in self.victims if self.op == "nr" or coins[j] < self.alpha]
        if not active:
            return intended.copy()
        oc = torch.as_tensor(np.concatenate(obs)).unsqueeze(0)
        base = torch.as_tensor(intended)
        r = base.clone()[active].requires_grad_(True)
        opt = torch.optim.Adam([r], lr=self.pgd_lr)
        for _ in range(self.pgd_steps):
            ex = base.clone()
            if self.op == "pr":
                ex[active] = r.clamp(-1, 1)
            else:
                ex[active] = (1 - self.alpha) * base[active] + self.alpha * r.clamp(-1, 1)
            loss = self.q(oc, ex.flatten().unsqueeze(0)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                r.clamp_(-1, 1)
        repl = intended.copy()
        repl[active] = r.detach().numpy()
        return repl

    def apply(self, obs, intended, draws):
        coins, rand = draws
        if self.alpha == 0:
            return intended
        if self.source == "R":
            repl = rand
        elif self.source == "G":
            repl = self._worst_case(obs, intended, coins)
        else:
            repl = intended.copy()
            with torch.no_grad():
                out = self.adv(torch.as_tensor(np.concatenate(obs)).unsqueeze(0))[0].numpy()
            repl[self.victims] = out.reshape(len(self.victims), 2)
        return self._execute(intended, repl, coins)


def run_episode(env, team, attack, seed, pert_seed):
    """Team return of one episode under an attack (attack=None means clean)."""
    rng = np.random.default_rng(pert_seed)
    obs = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        a = team(obs)
        draws = attack.draw(rng) if attack else None
        if attack:
            a = attack.apply(obs, a, draws)
        obs, _, team_r, done = env.step(a)
        total += team_r
    return total


def load_team(run_dir, n):
    from marl.nets import load_actor
    actors = [load_actor(f"{run_dir}/actor_{i}.pt") for i in range(n)]

    def team(obs):
        return np.stack([p(o) for p, o in zip(actors, obs)]).astype(np.float32)
    return team
