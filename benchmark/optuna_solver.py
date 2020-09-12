import argparse
import optuna

from cmaes import SepCMA
from kurobako import solver
from kurobako.solver.optuna import OptunaSolverFactory

parser = argparse.ArgumentParser()
parser.add_argument("sampler", choices=["cmaes", "sep-cmaes", "ipop-cmaes", "pycma"])
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


def create_cmaes_study(seed):
    sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=True)
    return optuna.create_study(sampler=sampler)


def create_sep_cmaes_study(seed):
    optuna.samplers._cmaes.CMA = SepCMA  # monkey patch
    sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=True)
    return optuna.create_study(sampler=sampler)


def create_ipop_cmaes_study(seed):
    sampler = optuna.samplers.CmaEsSampler(
        seed=seed,
        warn_independent_sampling=True,
        restart_strategy="ipop",
        inc_popsize=2,
    )
    return optuna.create_study(sampler=sampler)


def create_pycma_study(seed):
    sampler = optuna.integration.PyCmaSampler(
        seed=seed,
        warn_independent_sampling=True,
    )
    return optuna.create_study(sampler=sampler)


if __name__ == "__main__":
    if args.sampler == "cmaes":
        factory = OptunaSolverFactory(create_cmaes_study)
    elif args.sampler == "sep-cmaes":
        factory = OptunaSolverFactory(create_sep_cmaes_study)
    elif args.sampler == "ipop-cmaes":
        factory = OptunaSolverFactory(create_ipop_cmaes_study)
    elif args.sampler == "pycma":
        factory = OptunaSolverFactory(create_pycma_study)
    else:
        raise ValueError("unsupported sampler")

    runner = solver.SolverRunner(factory)
    runner.run()
