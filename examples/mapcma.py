import numpy as np
from cmaes import MAPCMA


def rosenbrock(x):
    dim = len(x)
    if dim < 2:
        raise ValueError("dimension must be greater one")
    return sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)


if __name__ == "__main__":
    dim = 20
    optimizer = MAPCMA(mean=np.zeros(dim), sigma=0.5, momentum_r=dim)
    print(" evals    f(x)")
    print("======  ==========")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = rosenbrock(x)
            evals += 1
            solutions.append((x, value))
            if evals % 1000 == 0:
                print(f"{evals:5d}  {value:10.5f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break
