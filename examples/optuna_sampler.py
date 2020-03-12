import optuna

from cmaes.sampler import CMASampler


def objective(trial: optuna.Trial):
    x1 = trial.suggest_uniform("x1", -4, 4)
    x2 = trial.suggest_uniform("x2", -4, 4)
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def main():
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = CMASampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=50, gc_after_trial=False)


if __name__ == "__main__":
    main()
