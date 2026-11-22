"""Residual-corrected actors on a frozen Phase 0 base (Phase 1, Stage 1).

ResidualActor (per agent):
    u_b   = pre-tanh output of the frozen base actor (Phase 0 checkpoint)
    x_t   = [o_t, a_{t-1}, a~_{t-1}, mu_b(o_t)]       a = own command, a~ = own executed action
    h_t   = GRU(relu(W x_t), h_{t-1})
    mu_t  = tanh(u_b + g_t * beta * tanh(D_t))      D_t (zero-initialised), g_t = sigmoid(.)
The residual is added before the squash (framework doc, section 3). D's head is zero-initialised,
so the combined actor equals the base exactly until training moves it. In MPE the previous
executed action is recoverable exactly from two own-velocity observations (except on collision
steps), so giving a~_{t-1} directly is equivalent to inferring it from motion.

FTActor (control FT): a copy of the base MLP with all weights trained, no memory, no evidence.

Stage 2 kinds (same interface):
    rcfast  ResidualActor with explicit evidence: e = (a~_{t-1} - a_{t-1}) / 0.3 and its running
            maximum norm m_t join the GRU input, and m_t also feeds the gate directly.
            The state carried between steps is [h, m] (state_dim = hidden + 1).
    rec     REC control: the ResidualActor network with the base weights trained too (no
            frozen base, so no base/residual split); initialised equal to the base.
    resmlp  RES-MLP control: residual head and gate on an MLP over [o_t, mu_b] only, no memory and
            no evidence inputs.

Corruption sources (all agents are victims in Stage 1 training):
    R     r_t ~ U([-1,1]^2) i.i.d.
    Rphi  r_t = clip(phi r_{t-1} + sqrt(1 - phi^2) u_t)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.nets import Actor


def load_base(path):
    d = torch.load(path, map_location="cpu")
    a = Actor(d["obs_dim"], d["act_dim"], d["hidden"])
    a.load_state_dict(d["state"])
    for p in a.parameters():
        p.requires_grad_(False)
    a.eval()
    return a


def base_pre(base, x):
    return base.l3(F.relu(base.l2(F.relu(base.l1(x)))))


class ResidualActor(nn.Module):
    recurrent = True

    def __init__(self, base, hidden=64, gate_bias=0.0, fast=False, train_base=False,
                 space="pre", res_scale=1.0):
        super().__init__()
        self.base = base
        self.fast, self.train_base = fast, train_base
        # space "pre": Phase 1 form, mu = tanh(u_b + g beta tanh(D)).
        # space "action" (Phase 2, 2026-11-22): mu = clip(tanh(u_b) + g beta res_scale tanh(D)),
        # so every action in the box is reachable however saturated the base is (res_scale 2).
        assert space in ("pre", "action")
        self.space, self.res_scale = space, float(res_scale)
        d = base.obs_dim + 6 + (3 if fast else 0)
        self.inp = nn.Linear(d, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.delta = nn.Linear(hidden, 2)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.constant_(self.gate.bias, gate_bias)
        if fast:
            self.gate_m = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.gate_m.weight)
        if train_base:
            # REC: train the base weights too; frozen_base is the untouched copy used for the
            # clean-episode penalty and for mu_b in diagnostics
            self.frozen_base = copy.deepcopy(base)
            for p in self.base.parameters():
                p.requires_grad_(True)
            self.base.train()
        self.hidden = hidden
        self.state_dim = hidden + (1 if fast else 0)

    def trainable(self):
        if self.train_base:
            return [p for n, p in self.named_parameters() if not n.startswith("frozen_base.")]
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def step(self, obs, prev_cmd, prev_exe, h, beta):
        """obs (B, d), prev_* (B, 2), h (B, state_dim) -> mu, mu_b, gate, h.
        mu_b is the frozen base command (for REC: the untrained copy)."""
        u_b = base_pre(self.base, obs)
        ref = torch.tanh(base_pre(self.frozen_base, obs)) if self.train_base else torch.tanh(u_b)
        if self.fast:
            hh, m = h[:, :self.hidden], h[:, self.hidden:]
            e = (prev_exe - prev_cmd) / 0.3
            m = torch.maximum(m, e.norm(dim=-1, keepdim=True))
            x = torch.cat([obs, prev_cmd, prev_exe, torch.tanh(u_b).detach(), e, m], -1)
        else:
            hh = h
            x = torch.cat([obs, prev_cmd, prev_exe, torch.tanh(u_b).detach()], -1)
        hh = self.gru(F.relu(self.inp(x)), hh)
        gl = self.gate(hh) + (self.gate_m(m) if self.fast else 0.0)
        g = torch.sigmoid(gl)
        if self.space == "action":
            mu = (torch.tanh(u_b) + g * beta * self.res_scale * torch.tanh(self.delta(hh))).clamp(-1, 1)
        else:
            mu = torch.tanh(u_b + g * beta * torch.tanh(self.delta(hh)))
        h = torch.cat([hh, m], -1) if self.fast else hh
        return mu, ref, g, h

    def unroll(self, obs, prev_cmd, prev_exe, beta):
        """obs (B, T, d), prev_* (B, T, 2) -> mu (B, T, 2), mu_b, g."""
        B, T, _ = obs.shape
        h = obs.new_zeros(B, self.state_dim)
        mus, mubs, gs = [], [], []
        for t in range(T):
            mu, mu_b, g, h = self.step(obs[:, t], prev_cmd[:, t], prev_exe[:, t], h, beta)
            mus.append(mu)
            mubs.append(mu_b)
            gs.append(g)
        return torch.stack(mus, 1), torch.stack(mubs, 1), torch.stack(gs, 1)


class FTActor(nn.Module):
    recurrent = False

    def __init__(self, base):
        super().__init__()
        self.base = base  # frozen copy, used for the clean-episode penalty only
        self.net = copy.deepcopy(base)
        for p in self.net.parameters():
            p.requires_grad_(True)
        self.net.train()

    def trainable(self):
        return list(self.net.parameters())

    def step(self, obs, prev_cmd, prev_exe, h, beta):
        return self.net(obs), torch.tanh(base_pre(self.base, obs)), None, h

    def unroll(self, obs, prev_cmd, prev_exe, beta):
        mu = self.net(obs)
        return mu, torch.tanh(base_pre(self.base, obs)), None


class ResMLPActor(nn.Module):
    """RES-MLP control: residual and gate from an MLP on [o_t, mu_b]; no memory, no evidence."""
    recurrent = False

    def __init__(self, base, hidden=64, gate_bias=0.0):
        super().__init__()
        self.base = base
        self.l1 = nn.Linear(base.obs_dim + 2, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.delta = nn.Linear(hidden, 2)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.hidden = hidden

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def _fwd(self, obs, beta):
        u_b = base_pre(self.base, obs)
        mu_b = torch.tanh(u_b)
        z = F.relu(self.l2(F.relu(self.l1(torch.cat([obs, mu_b], -1)))))
        g = torch.sigmoid(self.gate(z))
        return torch.tanh(u_b + g * beta * torch.tanh(self.delta(z))), mu_b, g

    def step(self, obs, prev_cmd, prev_exe, h, beta):
        mu, mu_b, g = self._fwd(obs, beta)
        return mu, mu_b, g, h

    def unroll(self, obs, prev_cmd, prev_exe, beta):
        return self._fwd(obs, beta)


KINDS = ("residual", "ft", "rcfast", "rec", "resmlp")


def make_actor(kind, base, hidden=64, space="pre", res_scale=1.0):
    if kind == "residual":
        return ResidualActor(base, hidden, space=space, res_scale=res_scale)
    if kind == "rcfast":
        return ResidualActor(base, hidden, fast=True)
    if kind == "rec":
        return ResidualActor(base, hidden, train_base=True, space=space, res_scale=res_scale)
    if kind == "resmlp":
        return ResMLPActor(base, hidden)
    if kind == "ft":
        return FTActor(base)
    raise ValueError(kind)


def corruption_sequence(source, phi, T, n, rng):
    """(T, n, 2) replacement actions."""
    u = rng.uniform(-1, 1, (T, n, 2)).astype(np.float32)
    if source == "R":
        return u
    r = np.empty_like(u)
    r[0] = u[0]
    for t in range(1, T):
        r[t] = np.clip(phi * r[t - 1] + np.sqrt(1 - phi ** 2) * u[t], -1, 1)
    return r


class Team:
    """Numpy team for evaluation: reset() per episode, act(obs) -> commands, observe(cmd, exe)."""

    def __init__(self, actors, beta=1.0):
        self.actors, self.beta, self.n = actors, beta, len(actors)
        self.reset()

    def reset(self):
        self.h = [torch.zeros(1, getattr(a, "state_dim", getattr(a, "hidden", 1)))
                  for a in self.actors]
        self.prev_cmd = np.zeros((self.n, 2), np.float32)
        self.prev_exe = np.zeros((self.n, 2), np.float32)

    def act(self, obs):
        out = []
        with torch.no_grad():
            for i, a in enumerate(self.actors):
                t = lambda x: torch.as_tensor(np.asarray(x, np.float32)).unsqueeze(0)
                mu, _, _, self.h[i] = a.step(t(obs[i]), t(self.prev_cmd[i]), t(self.prev_exe[i]),
                                             self.h[i], self.beta)
                out.append(mu[0].numpy())
        return np.stack(out).astype(np.float32)

    def observe(self, cmd, exe):
        self.prev_cmd, self.prev_exe = cmd.copy(), exe.copy()


class StatelessTeam:
    """Wrap a Phase 0 numpy team (marl.attack.load_team) in the Team interface."""

    def __init__(self, team):
        self.team = team

    def reset(self):
        pass

    def act(self, obs):
        return self.team(obs)

    def observe(self, cmd, exe):
        pass


def load_team_any(run_dir, n):
    """Phase 0 run dir (actor_<i>.pt) or Stage 1 run dir (stage1.pt)."""
    import os
    ck = [f for f in ("stage2.pt", "stage1.pt") if os.path.exists(os.path.join(run_dir, f))]
    if ck:
        d = torch.load(os.path.join(run_dir, ck[0]), map_location="cpu")
        bdir = d["base_dir"]
        if not os.path.exists(bdir):  # checkpoint moved between machines
            bdir = os.path.join("runs", os.path.basename(bdir.rstrip("/")))
        actors = []
        for i in range(n):
            base = load_base(os.path.join(bdir, f"actor_{i}.pt"))
            a = make_actor(d["kind"], base, d["hidden"], d.get("space", "pre"),
                           d.get("res_scale", 1.0))
            a.load_state_dict(d["actors"][i])
            a.eval()
            actors.append(a)
        return Team(actors, beta=1.0)
    from marl.attack import load_team
    return StatelessTeam(load_team(run_dir, n))
