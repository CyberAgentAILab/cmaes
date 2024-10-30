import numpy as np
import pytest
from cmaes import CMA


@pytest.mark.freethreaded(threads=10, iterations=200)
def test_simple_optimization():
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    def quadratic(x1: float, x2: float) -> float:
        return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break
