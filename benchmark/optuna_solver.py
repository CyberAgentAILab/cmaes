import argparse
import optuna

from kurobako import solver
from kurobako.solver.optuna import OptunaSolverFactory

parser = argparse.ArgumentParser()
parser.add_argument("sampler", choices=["cmaes", "ipop", "pycma"])
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


class CMASolverFactory(OptunaSolverFactory):
    def specification(self):
        spec = super().specification()
        spec.name = "cmaes"
        return spec


class PyCMASolverFactory(OptunaSolverFactory):
    def specification(self):
        spec = super().specification()
        spec.name = "pycma"
        return spec


def create_cmaes_study(seed):
    sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=True,)
    return optuna.create_study(sampler=sampler)


def create_ipop_cmaes_study(seed):
    sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=True,)
    return optuna.create_study(sampler=sampler)


def create_pycma_study(seed):
    sampler = optuna.integration.PyCmaSampler(
        seed=seed, warn_independent_sampling=True,
    )
    return optuna.create_study(sampler=sampler)


if __name__ == "__main__":
    if args.sampler == "cmaes":
        factory = CMASolverFactory(create_cmaes_study)
    elif args.sampler == "ipop":
        factory = ConnectionAbortedError(create_ipop_cmaes_study)
    elif args.sampler == "pycma":
        factory = PyCMASolverFactory(create_pycma_study)
    else:
        raise ValueError("unsupported sampler")

    runner = solver.SolverRunner(factory)
    runner.run()
