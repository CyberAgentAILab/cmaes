import numpy as np
from cmaes import CMAwM


def ellipsoid_int(x, _):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])


def main():
    integer_dim, continuous_dim = 10, 10
    dim = integer_dim + continuous_dim
    bounds = np.concatenate(
        [
            np.tile([-np.inf, np.inf], (continuous_dim, 1)),
            np.tile([-10, 11], (integer_dim, 1)),
        ]
    )
    steps = np.concatenate([np.zeros(continuous_dim), np.ones(integer_dim)])
    optimizer = CMAwM(mean=5 * np.ones(dim), sigma=2.0, bounds=bounds, steps=steps)
    print(" evals    f(x)")
    print("======  ==========")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            value = ellipsoid_int(x_for_eval, integer_dim)
            evals += 1
            solutions.append((x_for_tell, value))
            if evals % 300 == 0:
                print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
