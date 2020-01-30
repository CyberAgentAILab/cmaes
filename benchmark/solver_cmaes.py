import argparse
import optuna

from kurobako import solver
from kurobako.solver.optuna import OptunaSolverFactory

from cmaes.sampler import CMASampler

parser = argparse.ArgumentParser()
parser.add_argument("--startup", type=int, default=1)
parser.add_argument(
    "--loglevel", choices=["debug", "info", "warning", "error"], default="warning"
)
args = parser.parse_args()

if args.loglevel == "debug":
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
elif args.loglevel == "info":
    optuna.logging.set_verbosity(optuna.logging.INFO)
elif args.loglevel == "warning":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
elif args.loglevel == "error":
    optuna.logging.set_verbosity(optuna.logging.ERROR)


def create_study(seed):
    sampler = CMASampler(seed=seed, n_startup_trials=args.startup)
    return optuna.create_study(sampler=sampler)


if __name__ == "__main__":
    factory = OptunaSolverFactory(create_study)
    runner = solver.SolverRunner(factory)
    runner.run()
