import numpy as np

from cmaes import CMA
from cmaes import get_starting_point


def sphere(x1: float, x2: float, b: float) -> float:
    return (x1 - b) ** 2 + (x2 - b) ** 2


def main() -> None:
    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")

    seed = 0
    source_offset = 0.4
    target_offset = 0.6
    rng = np.random.RandomState(seed)

    # Generate solutions from a source task
    source_solutions = []
    for _ in range(1000):
        x = rng.random(2)
        value = sphere(x[0], x[1], source_offset)
        source_solutions.append((x, value))

    # Estimate a promising distribution of the source task
    ws_mean, ws_sigma, ws_cov = get_starting_point(
        source_solutions, gamma=0.1, alpha=0.1
    )
    optimizer = CMA(
        mean=ws_mean, sigma=ws_sigma, cov=ws_cov, population_size=8, seed=seed
    )

    # Run WS-CMA-ES
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = sphere(x[0], x[1], target_offset)
            solutions.append((x, value))
            print(
                f"{optimizer.generation:3d}  {value:10.5f}"
                f"  {x[0]:6.2f}  {x[1]:6.2f}"
            )
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
