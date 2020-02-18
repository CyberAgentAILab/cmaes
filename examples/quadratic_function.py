import numpy as np
from cmaes import CMA


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def main():
    cma_es = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(50):
        solutions = []
        for _ in range(cma_es.population_size):
            x = cma_es.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print("#{g} {value} (x1={x1}, x2 = {x2})".format(
                g=generation, value=value, x1=x[0], x2=x[1],
            ))
        cma_es.tell(solutions)


if __name__ == "__main__":
    main()
