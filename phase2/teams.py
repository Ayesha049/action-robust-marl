"""Load any team for Phase 2 evaluation in the Team interface (reset / act / observe).

Phase 0 run dirs (actor_<i>.pt) give a stateless team; Phase 1/2 run dirs (stage1.pt or
stage2.pt) give the recurrent team of phase1.residual. A Phase 2 RC-blind checkpoint
(blind=True) only ever sees its own command in place of the executed action.
"""

import os

import torch

from phase1.residual import load_team_any


class BlindTeam:
    def __init__(self, team):
        self.team = team

    def reset(self):
        self.team.reset()

    def act(self, obs):
        return self.team.act(obs)

    def observe(self, cmd, exe):
        self.team.observe(cmd, cmd)


def load_any(run_dir, n):
    team = load_team_any(run_dir, n)
    ck = os.path.join(run_dir, "stage1.pt")
    if os.path.exists(ck) and torch.load(ck, map_location="cpu").get("blind", False):
        team = BlindTeam(team)
    return team
