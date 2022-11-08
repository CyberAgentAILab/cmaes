import numpy as np
from cmaes import CMAwM


def ellipsoid_onemax(x, n_zdim):
    n = len(x)
    n_rdim = n - n_zdim
    r = 10
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    ellipsoid = sum([(1000 ** (i / (n_rdim - 1)) * x[i]) ** 2 for i in range(n_rdim)])
    onemax = n_zdim - (0.0 < x[(n - n_zdim) :]).sum()
    return ellipsoid + r * onemax


def main():
    binary_dim, continuous_dim = 10, 10
    dim = binary_dim + continuous_dim
    bounds = np.concatenate(
        [
            np.tile([-np.inf, np.inf], (continuous_dim, 1)),
            np.tile([0, 1], (binary_dim, 1)),
        ]
    )
    steps = np.concatenate([np.zeros(continuous_dim), np.ones(binary_dim)])
    optimizer = CMAwM(mean=np.zeros(dim), sigma=2.0, bounds=bounds, steps=steps)
    print(" evals    f(x)")
    print("======  ==========")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            value = ellipsoid_onemax(x_for_eval, binary_dim)
            evals += 1
            solutions.append((x_for_tell, value))
            if evals % 300 == 0:
                print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
