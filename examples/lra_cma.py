import numpy as np
from cmaes import CMA


def rastrigin(x):
    dim = len(x)
    if dim < 2:
        raise ValueError("dimension must be greater one")
    return 10 * dim + sum(x**2 - 10 * np.cos(2 * np.pi * x))


if __name__ == "__main__":
    dim = 40
    optimizer = CMA(mean=3 * np.ones(dim), sigma=2.0, seed=10, lr_adapt=True)

    for generation in range(50000):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = rastrigin(x)
            if generation % 500 == 0:
                print(f"#{generation} {value}")
            solutions.append((x, value))
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break
