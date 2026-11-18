import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Deterministic policy obs -> action in [-1,1]^act_dim."""

    def __init__(self, obs_dim, act_dim=2, hidden=64):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, hidden
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, act_dim)

    def forward(self, x):
        return torch.tanh(self.l3(F.relu(self.l2(F.relu(self.l1(x))))))


class Critic(nn.Module):
    """Centralised Q(all obs, all actions)."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, obs_cat, act_cat):
        x = torch.cat([obs_cat, act_cat], -1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))


class DiscreteQ(nn.Module):
    """Q over a small discrete set (MARTA switcher)."""

    def __init__(self, in_dim, n_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class CLUB(nn.Module):
    """CLUB upper bound on I(X; Y) with a Gaussian q(y|x) (Cheng et al. 2020).

    Same as CLUBContinuous in the MIR3 release (DIG-Beihang/MIR3, agents/nets/club.py):
    two one-hidden-layer nets of width hidden//2 give mu and a tanh-bounded log-variance.
    """

    def __init__(self, x_dim, y_dim, hidden=64):
        super().__init__()
        h = max(hidden // 2, 1)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, h), nn.ReLU(), nn.Linear(h, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, h), nn.ReLU(), nn.Linear(h, y_dim), nn.Tanh())

    def forward(self, x, y):
        """Per-sample bound: log q(y_i|x_i) - mean_j log q(y_j|x_i) (constants cancel)."""
        mu, logvar = self.p_mu(x), self.p_logvar(x)
        pos = -(mu - y) ** 2 / 2.0 / logvar.exp()
        neg = -((y.unsqueeze(0) - mu.unsqueeze(1)) ** 2).mean(1) / 2.0 / logvar.exp()
        return pos.sum(1) - neg.sum(1)

    def learning_loss(self, x, y):
        mu, logvar = self.p_mu(x), self.p_logvar(x)
        return ((mu - y) ** 2 / logvar.exp() + logvar).sum(1).mean()


def save_actor(actor, path):
    torch.save({"obs_dim": actor.obs_dim, "act_dim": actor.act_dim, "hidden": actor.hidden,
                "state": actor.state_dict()}, path)


def load_actor(path):
    """Return a numpy callable obs -> action from a saved actor."""
    d = torch.load(path, map_location="cpu")
    a = Actor(d["obs_dim"], d["act_dim"], d["hidden"])
    a.load_state_dict(d["state"])
    a.eval()

    def policy(obs):
        with torch.no_grad():
            return a(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
    return policy


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(tau * p)
