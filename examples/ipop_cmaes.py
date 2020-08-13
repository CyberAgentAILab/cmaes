import math
import numpy as np
from cmaes import CMA


def ackley(x1, x2):
    return (
        -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))
        - math.exp(0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2)))
        + math.e
        + 20
    )


def main():
    seed = 0
    rng = np.random.RandomState(1)

    bounds = np.array([[-32.768, 32.768], [-32.768, 32.768]])
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (rng.rand(2) * (upper_bounds - lower_bounds))
    sigma = 32.768 * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    # Multiplier for increasing population size before each restart.
    inc_popsize = 2

    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")
    for generation in range(200):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = ackley(x[0], x[1])
            solutions.append((x, value))
            print(f"{generation:3d}  {value:10.5f}  {x[0]:6.2f}  {x[1]:6.2f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            seed += 1
            popsize = optimizer.population_size * inc_popsize
            mean = lower_bounds + (rng.rand(2) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                seed=seed,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={}".format(popsize))


if __name__ == "__main__":
    main()
