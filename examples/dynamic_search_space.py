import optuna

from cmaes.sampler import CMASampler


def objective(trial):
    x = trial.suggest_uniform("x", -100, 100)
    if trial.number < 20:
        y = trial.suggest_int("y", -10, 0)
    else:
        y = trial.suggest_int("y", 1, 10)
    z = trial.suggest_uniform("z", -100, 100)
    return x ** 2 + y + z


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(sampler=CMASampler())
    study.optimize(objective, n_trials=40)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
