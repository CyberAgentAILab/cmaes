import numpy as np
from cmaes import CMA


def ellipsoid(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])


def main():
    dim = 40
    optimizer = CMA(mean=3 * np.ones(dim), sigma=2.0)
    print(" evals    f(x)")
    print("======  ==========")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = ellipsoid(x)
            evals += 1
            solutions.append((x, value))
            if evals % 3000 == 0:
                print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
