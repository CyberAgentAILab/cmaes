import optuna

from cmaes.sampler import CMASampler


def objective(trial):
    x = trial.suggest_uniform("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x ** 2 + y


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(sampler=CMASampler())
    study.optimize(objective, n_trials=40)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
