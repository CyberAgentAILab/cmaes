import argparse
import logging
import time
import optuna
import numpy as np

from optuna.integration.cma import CmaEsSampler
from cmaes.sampler import CMASampler

parser = argparse.ArgumentParser()
parser.add_argument("--times", type=int, default=1)
args = parser.parse_args()


def run_optimize(sampler_str, storage_str, trials, params, index):
    storage = None
    if storage_str == "sqlite":
        storage = f"sqlite:///db_{sampler_str}_{params}_{trials}_{index}.sqlite3"

    if sampler_str == "pycma":
        sampler = CmaEsSampler()
    else:
        sampler = CMASampler()
    study = optuna.create_study(sampler=sampler, storage=storage)

    def objective(trial: optuna.Trial):
        val = 0
        for i in range(params):
            xi = trial.suggest_uniform(str(i), -4, 4)
            val += (xi - 2) ** 2
        return val

    start = time.time()
    study.optimize(objective, n_trials=trials, gc_after_trial=False)
    return time.time() - start


def print_markdown_table(results):
    print(f"The benchmark is executed {args.times} times.")
    print("")
    # Header
    print("| trials/params | storage | pycma's sampler | this library |")
    print("| ------------- | ------- | --------------- | ------------ |")

    for trials_params, storage, pycma_score, cmaes_score in results:
        print(f"| {trials_params} | {storage} | {pycma_score} | {cmaes_score} |")


def main():
    results = []
    for storage, trials, params in [
        ("memory", 100, 5),
        ("sqlite", 100, 5),
        ("memory", 500, 5),
        ("sqlite", 500, 5),
        ("memory", 500, 50),
        ("sqlite", 500, 50),
    ]:
        print("Started:", storage, trials, params)

        elapsed = []
        for i in range(args.times):
            elapsed.append(run_optimize("pycma", storage, trials, params, i))
        pycma_score = f"{np.mean(elapsed):.3f} (+/- {np.std(elapsed):.3f})s"

        elapsed = []
        for i in range(args.times):
            elapsed.append(run_optimize("cmaes", storage, trials, params, i))
        cmaes_score = f"{np.mean(elapsed):7.3f} (+/- {np.std(elapsed):.3f})s"

        results.append((f"{trials:3} / {params:3}", storage, pycma_score, cmaes_score))

    print_markdown_table(results)


if __name__ == "__main__":
    logging.disable(level=logging.INFO)

    main()
