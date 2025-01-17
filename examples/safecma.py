import numpy as np
from cmaes import SafeCMA


def example1():
    """
    example with a single safety function
    """

    # number of dimensions
    dim = 5

    # objective function
    def quadratic(x):
        coef = 1000 ** (np.arange(dim) / float(dim - 1))
        return np.sum((x * coef) ** 2)

    # safety function
    def safe_function(x):
        return x[0]

    # safe seeds
    safe_seeds_num = 10
    safe_seeds = (np.random.rand(safe_seeds_num, dim) * 2 - 1) * 5
    safe_seeds[:, 0] = -np.abs(safe_seeds[:, 0])

    # evaluation of safe seeds (with a single safety function)
    seeds_evals = np.array([quadratic(x) for x in safe_seeds])
    seeds_safe_evals = np.stack([[safe_function(x)] for x in safe_seeds])
    safety_threshold = np.array([0])

    # optimizer (safe CMA-ES)
    optimizer = SafeCMA(
        sigma=1.0,
        safety_threshold=safety_threshold,
        safe_seeds=safe_seeds,
        seeds_evals=seeds_evals,
        seeds_safe_evals=seeds_safe_evals,
    )

    unsafe_eval_counts = 0
    best_eval = np.inf

    for generation in range(400):
        solutions = []
        for _ in range(optimizer.population_size):
            # Ask a parameter
            x = optimizer.ask()
            value = quadratic(x)
            safe_value = np.array([safe_function(x)])

            # save best eval
            best_eval = np.min((best_eval, value))
            unsafe_eval_counts += safe_value > safety_threshold

            solutions.append((x, value, safe_value))

        # Tell evaluation values.
        optimizer.tell(solutions)

        print(f"#{generation} ({best_eval} {unsafe_eval_counts})")

        if optimizer.should_stop():
            break


def example2():
    """
    example with multiple safety functions
    """

    # number of dimensions
    dim = 5

    # objective function
    def quadratic(x):
        coef = 1000 ** (np.arange(dim) / float(dim - 1))
        return np.sum((x * coef) ** 2)

    # safety functions
    def safe_function1(x):
        return x[0]

    def safe_function2(x):
        return x[1]

    # safe seeds
    safe_seeds_num = 10
    safe_seeds = (np.random.rand(safe_seeds_num, dim) * 2 - 1) * 5
    safe_seeds[:, 0] = -np.abs(safe_seeds[:, 0])
    safe_seeds[:, 1] = -np.abs(safe_seeds[:, 1])

    # evaluation of safe seeds (with multiple safety functions)
    seeds_evals = np.array([quadratic(x) for x in safe_seeds])
    seeds_safe_evals = np.stack(
        [[safe_function1(x), safe_function2(x)] for x in safe_seeds]
    )
    safety_threshold = np.array([0, 0])

    # optimizer (safe CMA-ES)
    optimizer = SafeCMA(
        sigma=1.0,
        safety_threshold=safety_threshold,
        safe_seeds=safe_seeds,
        seeds_evals=seeds_evals,
        seeds_safe_evals=seeds_safe_evals,
    )

    unsafe_eval_counts = 0
    best_eval = np.inf

    for generation in range(400):
        solutions = []
        for _ in range(optimizer.population_size):
            # Ask a parameter
            x = optimizer.ask()
            value = quadratic(x)
            safe_value = np.array([safe_function1(x), safe_function2(x)])

            # save best eval
            best_eval = np.min((best_eval, value))
            unsafe_eval_counts += safe_value > safety_threshold

            solutions.append((x, value, safe_value))

        # Tell evaluation values.
        optimizer.tell(solutions)

        print(f"#{generation} ({best_eval} {unsafe_eval_counts})")

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    example1()
    example2()
