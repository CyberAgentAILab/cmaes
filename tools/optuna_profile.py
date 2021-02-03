import argparse
import cProfile
import logging
import pstats
import optuna

parser = argparse.ArgumentParser()
parser.add_argument("--storage", choices=["memory", "sqlite"], default="memory")
parser.add_argument("--params", type=int, default=100)
parser.add_argument("--trials", type=int, default=1000)
args = parser.parse_args()


def objective(trial: optuna.Trial):
    val = 0
    for i in range(args.params):
        xi = trial.suggest_uniform(str(i), -4, 4)
        val += (xi - 2) ** 2
    return val


def main():
    logging.disable(level=logging.INFO)
    storage = None
    if args.storage == "sqlite":
        storage = f"sqlite:///db-{args.trials}-{args.params}.sqlite3"
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler, storage=storage)

    profiler = cProfile.Profile()
    profiler.runcall(
        study.optimize, objective, n_trials=args.trials, gc_after_trial=False
    )
    profiler.dump_stats("profile.stats")

    stats = pstats.Stats("profile.stats")
    stats.sort_stats("time").print_stats(5)


if __name__ == "__main__":
    main()
