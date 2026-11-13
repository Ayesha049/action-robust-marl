"""Job lists for the Phase 0c follow-up (printed as shell commands, one per line).

Two experiments, split into two machine groups:

  dose  (spread, seeds 0-4): does more M3DDPG / ERNIE-A-c regularisation buy PR/NR
        robustness?  Settings stronger than the Phase 0b winners are added. The Phase 0b
        winners themselves (m3 eps 0.1, ernie lam 1.0 eps 0.05) already exist at seeds 0-4
        in runs/ and runs_ernie/, as does the MADDPG reference.
  val   (tag, full self-play, seeds 0-4): does our M3DDPG reproduce its own paper's claim
        (Li et al. 2019, Fig. 2 cross-play and Fig. 3 trained adversary)?

Group a: dose M3DDPG + val.   Group b: dose ERNIE-A-c.

  python phase0c.py train --group a | scripts/run_local.sh 30
  python phase0c.py post  --group a | scripts/run_local.sh 30
"""

import argparse
import os

SEEDS = [0, 1, 2, 3, 4]
STEPS = 1_500_000

DOSE_M3 = {f"m3_eps{e}": f"--method m3ddpg --m3-eps {e}" for e in ("0.001", "0.01", "0.3", "1.0")}
DOSE_ERNIE = {
    "ernie_l0.1_e0.05": "--method ernie --ernie-lam 0.1 --ernie-eps 0.05",
    "ernie_l0.1_e0.2": "--method ernie --ernie-lam 0.1 --ernie-eps 0.2",
    "ernie_l1_e0.2": "--method ernie --ernie-lam 1.0 --ernie-eps 0.2",
    "ernie_l1_e0.5": "--method ernie --ernie-lam 1.0 --ernie-eps 0.5",
    "ernie_l10_e0.2": "--method ernie --ernie-lam 10.0 --ernie-eps 0.2",
}
# tag, full self-play (all four agents learn). "ours" = the implementation used in Phase 0
# (perturbation in critic target and actor loss); "critic" = released-code behaviour
# (critic target only).
VAL = {
    "ma": "--method maddpg",
    "m3_ours_e0.001": "--method m3ddpg --m3-eps 0.001",
    "m3_ours_e0.01": "--method m3ddpg --m3-eps 0.01",
    "m3_ours_e0.1": "--method m3ddpg --m3-eps 0.1",
    "m3_critic_e0.001": "--method m3ddpg --m3-eps 0.001 --m3-actor false",
    "m3_critic_e0.1": "--method m3ddpg --m3-eps 0.1 --m3-actor false",
}
ADV_CELLS = [("pr", "0.3", "all"), ("nr", "0.3", "all"), ("pr", "1.0", "one")]
ADV_SEEDS = [0, 1]
PREY_SEEDS = [0, 1]


def method_of(flags):
    return flags.split("--method ")[1].split()[0]


def train_jobs(group):
    jobs = []
    if group == "a":
        for cfg, fl in DOSE_M3.items():
            jobs += [("runs_dose", cfg, "spread", "", fl, s) for s in SEEDS]
        for cfg, fl in VAL.items():
            jobs += [("runs_val", cfg, "tag", "--mode all", fl, s) for s in SEEDS]
    else:
        for cfg, fl in DOSE_ERNIE.items():
            jobs += [("runs_dose", cfg, "spread", "", fl, s) for s in SEEDS]
    for root, cfg, task, mode, fl, s in jobs:
        d = f"{root}/{cfg}/{task}__{method_of(fl)}__s{s}"
        if os.path.exists(f"{d}/DONE"):
            continue
        print(f"mkdir -p {d} && python train.py --task {task} {mode} --seed {s} --steps {STEPS} "
              f"--out-dir {root}/{cfg} {fl} > {d}/train.out 2>&1")


def post_jobs(group):
    dirs = []
    if group == "a":
        dirs += [("dose", f"runs_dose/{c}") for c in DOSE_M3]
        dirs += [("val", f"runs_val/{c}") for c in VAL]
    else:
        dirs += [("dose", f"runs_dose/{c}") for c in DOSE_ERNIE]
    long_jobs, short_jobs = [], []
    for kind, root in dirs:
        for s in SEEDS:
            ds = [x for x in (f"{root}/{t}__{m}__s{s}" for t in ("spread", "tag")
                              for m in ("m3ddpg", "ernie", "maddpg")) if os.path.exists(f"{x}/DONE")]
            for d in ds:
                if kind == "dose":
                    for op, al, sc in ADV_CELLS:
                        for as_ in ADV_SEEDS:
                            if not os.path.exists(f"{d}/adv/{op}_a{al}_{sc}_s{as_}.json"):
                                long_jobs.append(f"python train_adversary.py --run-dir {d} --op {op} "
                                                 f"--alpha {al} --scope {sc} --seed {as_} "
                                                 f"> /dev/null 2>> {d}/adv_errors.log")
                    if not os.path.exists(f"{d}/eval_curves.csv"):
                        short_jobs.append(f"( [ -f {d}/eval_critic.pt ] || python eval_critic.py "
                                          f"--run-dir {d} ) > {d}/eval.out 2>&1 && python "
                                          f"eval_curves.py --run-dir {d} >> {d}/eval.out 2>&1")
                else:
                    for ps in PREY_SEEDS:
                        if not os.path.exists(f"{d}/prey_s{ps}.json"):
                            long_jobs.append(f"python train_prey.py --run-dir {d} --seed {ps} "
                                             f"> {d}/prey_s{ps}.out 2>&1")
    for j in long_jobs + short_jobs:  # long jobs first so the pool drains evenly
        print(j)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["train", "post"])
    p.add_argument("--group", choices=["a", "b"], required=True)
    a = p.parse_args()
    (train_jobs if a.stage == "train" else post_jobs)(a.group)
