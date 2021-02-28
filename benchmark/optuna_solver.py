import argparse
import optuna
import warnings

from kurobako import solver
from kurobako.solver.optuna import OptunaSolverFactory

warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    module="optuna.samplers._cmaes",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "sampler",
    choices=["cmaes", "sep-cmaes", "ipop-cmaes", "ipop-sep-cmaes", "pycma", "ws-cmaes"],
)
parser.add_argument(
    "--loglevel", choices=["debug", "info", "warning", "error"], default="warning"
)
parser.add_argument("--warm-starting-trials", type=int, default=0)
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
    return optuna.create_study(sampler=sampler, pruner=optuna.pruners.NopPruner())


def create_sep_cmaes_study(seed):
    sampler = optuna.samplers.CmaEsSampler(
        seed=seed, warn_independent_sampling=True, use_separable_cma=True
    )
    return optuna.create_study(sampler=sampler, pruner=optuna.pruners.NopPruner())


def create_ipop_cmaes_study(seed):
    sampler = optuna.samplers.CmaEsSampler(
        seed=seed,
        warn_independent_sampling=True,
        restart_strategy="ipop",
        inc_popsize=2,
    )
    return optuna.create_study(sampler=sampler, pruner=optuna.pruners.NopPruner())


def create_ipop_sep_cmaes_study(seed):
    sampler = optuna.samplers.CmaEsSampler(
        seed=seed,
        warn_independent_sampling=True,
        restart_strategy="ipop",
        inc_popsize=2,
        use_separable_cma=True,
    )
    return optuna.create_study(sampler=sampler, pruner=optuna.pruners.NopPruner())


def create_pycma_study(seed):
    sampler = optuna.integration.PyCmaSampler(
        seed=seed,
        warn_independent_sampling=True,
    )
    return optuna.create_study(sampler=sampler, pruner=optuna.pruners.NopPruner())


class WarmStartingCmaEsSampler(optuna.samplers.BaseSampler):
    def __init__(self, seed, warm_starting_trials: int) -> None:
        self._seed = seed
        self._warm_starting = True
        self._warm_starting_trials = warm_starting_trials
        self._sampler = optuna.samplers.RandomSampler(seed=seed)
        self._source_trials = []

    def infer_relative_search_space(self, study, trial):
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study,
        trial,
        search_space,
    ):
        return self._sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def after_trial(
        self,
        study,
        trial,
        state,
        values,
    ):
        if not self._warm_starting:
            return self._sampler.after_trial(study, trial, state, values)

        if len(self._source_trials) < self._warm_starting_trials:
            assert state == optuna.trial.TrialState.PRUNED

            self._source_trials.append(
                optuna.create_trial(
                    params=trial.params,
                    distributions=trial.distributions,
                    values=values,
                )
            )
        if len(self._source_trials) == self._warm_starting_trials:
            self._sampler = optuna.samplers.CmaEsSampler(
                seed=self._seed + 1, source_trials=self._source_trials or None
            )
            self._warm_starting = False
        else:
            return self._sampler.after_trial(study, trial, state, values)


def create_warm_start_study(seed):
    sampler = WarmStartingCmaEsSampler(seed, args.warm_starting_trials)
    return optuna.create_study(sampler=sampler, pruner=optuna.pruners.NopPruner())


if __name__ == "__main__":
    if args.sampler == "cmaes":
        factory = OptunaSolverFactory(create_cmaes_study)
    elif args.sampler == "sep-cmaes":
        factory = OptunaSolverFactory(create_sep_cmaes_study)
    elif args.sampler == "ipop-cmaes":
        factory = OptunaSolverFactory(create_ipop_cmaes_study)
    elif args.sampler == "ipop-sep-cmaes":
        factory = OptunaSolverFactory(create_ipop_sep_cmaes_study)
    elif args.sampler == "pycma":
        factory = OptunaSolverFactory(create_pycma_study)
    elif args.sampler == "ws-cmaes":
        factory = OptunaSolverFactory(
            create_warm_start_study, warm_starting_trials=args.warm_starting_trials
        )
    else:
        raise ValueError("unsupported sampler")

    runner = solver.SolverRunner(factory)
    runner.run()
