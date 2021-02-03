import numpy as np
from cmaes import CMA, get_warm_start_mgd


def sphere(x1: float, x2: float, b: float) -> float:
    return (x1 - b) ** 2 + (x2 - b) ** 2


def main() -> None:
    source_offset, target_offset = 0.4, 0.6

    # Generate solutions from a source task
    source_solutions = []
    for _ in range(1000):
        x = np.random.random(2)
        value = sphere(x[0], x[1], source_offset)
        source_solutions.append((x, value))

    # Estimate a promising distribution of the source task
    ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(
        source_solutions, gamma=0.1, alpha=0.1
    )
    optimizer = CMA(mean=ws_mean, sigma=ws_sigma, cov=ws_cov)

    # Run WS-CMA-ES
    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")
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
