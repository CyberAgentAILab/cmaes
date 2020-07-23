import numpy as np
from cmaes import CMA


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def main():
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)
    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))

            msg = "{g:3d}  {value:10.5f}  {x1:6.2f}  {x2:6.2f}".format(
                g=generation, value=value, x1=x[0], x2=x[1],
            )
            print(msg)
        optimizer.tell(solutions)


if __name__ == "__main__":
    main()
