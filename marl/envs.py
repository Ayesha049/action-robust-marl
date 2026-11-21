"""MPE (mpe2) tasks with a true 2-D continuous force action per agent.

mpe2 with continuous_actions=True takes a 5-vector a in [0,1]^5 and applies
u = (a[2] - a[1], a[4] - a[3]) * sensitivity. We expose f in [-1,1]^2 and map
it exactly: a = [0, (-f_x)+, (f_x)+, (-f_y)+, (f_y)+].

Tasks
-----
spread : simple_spread_v3, N=3 cooperative agents. Team = all agents.
transport : VMAS transport (vmas 1.5.2), 4 agents push one package (mass 50) to a goal.
         Fixed horizon of TRANSPORT_STEPS; VMAS's done (package on goal) is ignored, as reward is
         0 while the package stays on the goal. Action f in [-1,1]^2 is passed as the VMAS force
         (u_range 1, u_multiplier 0.6). Team reward = the shared shaped reward.
tag    : simple_tag_v3, 3 predators ("adversary_*") vs 1 prey ("agent_0").
         Team = predators. The prey is driven by a frozen policy (or controlled
         by the learner when mode="all", used only to pre-train the prey).
"""

import numpy as np
from mpe2 import simple_spread_v3, simple_tag_v3  # noqa: E402

MAX_CYCLES = 25


def force_to_mpe(f):
    f = np.clip(np.asarray(f, np.float32), -1.0, 1.0)
    return np.array([0.0, max(-f[0], 0.0), max(f[0], 0.0), max(-f[1], 0.0), max(f[1], 0.0)],
                    np.float32)


class TeamEnv:
    """Parallel MPE env exposing only the controlled agents.

    reset(seed) -> list of obs (one array per controlled agent)
    step(actions) -> obs, rewards (per controlled agent), team_reward, done
    """

    def __init__(self, task, mode="team", prey_policy=None):
        self.task = task
        if task == "spread":
            self.env = simple_spread_v3.parallel_env(N=3, max_cycles=MAX_CYCLES,
                                                     continuous_actions=True)
            self.team = [f"agent_{i}" for i in range(3)]
            self.others = []
        elif task == "tag":
            self.env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, num_obstacles=2,
                                                  max_cycles=MAX_CYCLES, continuous_actions=True)
            self.team = [f"adversary_{i}" for i in range(3)]
            self.others = ["agent_0"]
        else:
            raise ValueError(task)
        self.mode = mode
        self.controlled = self.team + (self.others if mode == "all" else [])
        self.prey_policy = prey_policy
        if task == "tag" and mode == "team" and prey_policy is None:
            raise ValueError("tag in team mode needs a frozen prey policy")
        self.n = len(self.controlled)
        self.n_team = len(self.team)
        self.obs_dims = [self.env.observation_space(a).shape[0] for a in self.controlled]
        self.act_dim = 2
        self._obs = None

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self._obs = obs
        self.t = 0
        return [obs[a].astype(np.float32) for a in self.controlled]

    def step(self, actions):
        act = {a: force_to_mpe(actions[i]) for i, a in enumerate(self.controlled)}
        for a in self.others:
            if a not in act:
                act[a] = force_to_mpe(self.prey_policy(self._obs[a]))
        obs, rew, term, trunc, _ = self.env.step(act)
        self._obs = obs
        self.t += 1
        done = self.t >= MAX_CYCLES or any(term.values()) or any(trunc.values())
        r = np.array([rew[a] for a in self.controlled], np.float32)
        team_r = float(np.mean([rew[a] for a in self.team]))
        return [obs[a].astype(np.float32) for a in self.controlled], r, team_r, done

    def close(self):
        self.env.close()


TRANSPORT_STEPS = 100


class VmasTeamEnv:
    """VMAS scenario with one environment, exposing the TeamEnv interface."""

    def __init__(self, scenario, n_agents=4, steps=TRANSPORT_STEPS):
        import torch
        import vmas
        self._torch = torch
        self.task, self.steps = scenario, steps
        self.env = vmas.make_env(scenario, num_envs=1, device="cpu", continuous_actions=True,
                                 seed=0, n_agents=n_agents)
        self.n = self.n_team = n_agents
        self.controlled = self.team = [f"agent_{i}" for i in range(n_agents)]
        self.obs_dims = [int(s.shape[0]) for s in self.env.observation_space]
        self.act_dim = 2
        self.t = 0

    def _np(self, obs):
        return [o[0].numpy().astype(np.float32) for o in obs]

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
        self.t = 0
        return self._np(self.env.reset())

    def step(self, actions):
        acts = [self._torch.as_tensor(np.clip(actions[i], -1, 1), dtype=self._torch.float32)
                .unsqueeze(0) for i in range(self.n)]
        obs, rews, _, _ = self.env.step(acts)
        self.t += 1
        r = np.array([float(x[0]) for x in rews], np.float32)
        return self._np(obs), r, float(r.mean()), self.t >= self.steps

    def close(self):
        pass


def make_env(task, mode="team", prey_path=None):
    if task == "transport":
        return VmasTeamEnv("transport")
    prey = None
    if task == "tag" and mode == "team":
        from marl.nets import load_actor
        prey = load_actor(prey_path or "assets/tag_prey.pt")
    return TeamEnv(task, mode=mode, prey_policy=prey)
