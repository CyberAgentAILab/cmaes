import math
import numpy as np
from cmaes import CMA


def ackley(x1, x2):
    return (
        -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1**2 + x2**2)))
        - math.exp(0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2)))
        + math.e
        + 20
    )


def main():
    seed = 0
    rng = np.random.RandomState(0)

    bounds = np.array([[-32.768, 32.768], [-32.768, 32.768]])
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (rng.rand(2) * (upper_bounds - lower_bounds))
    sigma0 = 32.768 * 2 / 5  # 1/5 of the domain width
    sigma = sigma0
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    inc_popsize = 2

    # Initial run is with "normal" population size; it is
    # the large population before first doubling, but its
    # budget accounting is the same as in case of small
    # population.
    poptype = "small"

    while n_restarts <= 5:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = ackley(x[0], x[1])
            solutions.append((x, value))
            # print("{:10.5f}  {:6.2f}  {:6.2f}".format(value, x[0], x[1]))
        optimizer.tell(solutions)

        if optimizer.should_stop():
            seed += 1
            n_eval = optimizer.population_size * optimizer.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize**n_restarts
                popsize = math.floor(
                    popsize0 * popsize_multiplier ** (rng.uniform() ** 2)
                )
                sigma = sigma0 * 10 ** (-2 * rng.uniform())
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize**n_restarts)
                sigma = sigma0
            mean = lower_bounds + (rng.rand(2) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                seed=seed,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={} ({})".format(popsize, poptype))


if __name__ == "__main__":
    main()
