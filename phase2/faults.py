"""Phase 2 persistent actuator faults (gain, bias, slow drift, stuck, rotation), with a random onset.

A fault cell is (kind, strength, victims). For each episode, sample(rng) draws the onset step
and the per-victim parameters. Before the onset nothing is corrupted. From the onset step t0:

  gain   exe_j = clip(g * cmd_j)                       g = strength
  bias   exe_j = clip(cmd_j + b_j)                     ||b_j|| = strength, direction random
                                                       (or fixed by `direction`, in eighths of a turn)
  drift  exe_j = clip(cmd_j + (t - t0 + 1) eta u_j)    eta = strength, u_j a random unit vector
  stuck  exe_j = s_j                                   s_j ~ U([-1,1]^2), held
  rot    exe_j = clip(R(sigma_j theta) cmd_j)          theta = strength in degrees, sigma_j = +-1
                                                       random (misaligned / mis-wired actuator)

Rotation keeps the force's magnitude, so unlike gain and bias it does not remove authority:
90 and 180 degrees map the action box onto itself and are always exactly restorable; other
angles lose only the part of a corner action that leaves the box (added 2026-11-21, after the
tag Stage 0 showed gain/bias faults on a saturating base are authority-limited).

Every draw comes from the episode's own generator in a fixed order, so all methods and oracle
predictors face the same onsets and parameters (common random numbers).
"""

import numpy as np

KINDS = ("gain", "bias", "drift", "stuck", "rot")
ONSET_MAX = 12  # onset t0 ~ U{0, ..., 12} in a 25-step spread episode


class Fault:
    def __init__(self, kind, strength, victims, n, direction=None):
        assert kind in KINDS
        self.kind, self.strength, self.victims, self.n = kind, float(strength), list(victims), n
        self.direction = direction

    def sample(self, rng):
        t0 = int(rng.integers(0, ONSET_MAX + 1))
        theta = rng.uniform(0, 2 * np.pi, self.n)
        stuck = rng.uniform(-1, 1, (self.n, 2)).astype(np.float32)
        if self.direction is not None:
            theta[:] = 2 * np.pi * self.direction / 8
        unit = np.stack([np.cos(theta), np.sin(theta)], -1).astype(np.float32)
        sign = np.where(rng.random(self.n) < 0.5, -1.0, 1.0)  # drawn last: earlier draws unchanged
        return {"t0": t0, "unit": unit, "stuck": stuck, "sign": sign}

    def rotation(self, ep, j, inverse=False):
        a = np.deg2rad(self.strength) * ep["sign"][j] * (-1 if inverse else 1)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]], np.float32)

    def params(self, ep, t):
        """(gain, bias) per victim at step t, as arrays (n,) and (n, 2); None before onset."""
        if t < ep["t0"]:
            return None
        g = np.ones(self.n, np.float32)
        b = np.zeros((self.n, 2), np.float32)
        for j in self.victims:
            if self.kind == "gain":
                g[j] = self.strength
            elif self.kind == "bias":
                b[j] = self.strength * ep["unit"][j]
            elif self.kind == "drift":
                b[j] = (t - ep["t0"] + 1) * self.strength * ep["unit"][j]
        return g, b

    def execute(self, ep, t, cmd):
        p = self.params(ep, t)
        if p is None:
            return cmd
        out = cmd.copy()
        g, b = p
        for j in self.victims:
            if self.kind == "stuck":
                out[j] = ep["stuck"][j]
            elif self.kind == "rot":
                out[j] = self.rotation(ep, j) @ cmd[j]
            else:
                out[j] = g[j] * cmd[j] + b[j]
        return np.clip(out, -1, 1)


def oracle_command(fault, ep, t, a_star, predictor):
    """Analytic correction a = clip((a* - b) / g), or clip(R^-1 a*) for rotation, from the
    fault's parameters.

    none     no correction
    exact    current parameters, from the onset (clairvoyant)
    d<k>     from step t0 + k on, with the previous step's parameters (what an agent can infer
             from its last executed action); identical to exact for gain and bias after t0 + k
    Returns (command, n_components_clipped, n_components_corrected).
    """
    if predictor == "none" or fault.kind == "stuck":
        return a_star, 0, 0
    if predictor == "exact":
        p = fault.params(ep, t)
    else:
        k = int(predictor[1:])
        p = fault.params(ep, t - 1) if t >= ep["t0"] + k else None
    if p is None:
        return a_star, 0, 0
    g, b = p
    cmd = a_star.copy()
    clipped = 0
    for j in fault.victims:
        if fault.kind == "rot":
            raw = fault.rotation(ep, j, inverse=True) @ a_star[j]
        else:
            raw = (a_star[j] - b[j]) / g[j]
        clipped += int((np.abs(raw) > 1).sum())
        cmd[j] = np.clip(raw, -1, 1)
    return cmd, clipped, 2 * len(fault.victims)
