import argparse
import logging
import optuna

from optuna.integration.cma import CmaEsSampler
from cmaes.sampler import CMASampler

parser = argparse.ArgumentParser()
parser.add_argument("--params", type=int, default=100)
parser.add_argument("--trials", type=int, default=1000)
parser.add_argument("--sampler", choices=["pycma", "cmaes"], default="cmaes")
args = parser.parse_args()


def objective(trial: optuna.Trial):
    val = 0
    for i in range(args.params):
        xi = trial.suggest_uniform(str(i), -4, 4)
        val += (xi - 2) ** 2
    return val


def main():
    logging.disable(level=logging.INFO)
    if args.sampler == "pycma":
        sampler = CmaEsSampler()
    else:
        sampler = CMASampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=args.trials, gc_after_trial=False)


if __name__ == "__main__":
    main()
