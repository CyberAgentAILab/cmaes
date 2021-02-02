import math
from typing import Callable

import numpy as np
from cmaes import CMA
from cmaes import get_starting_point

REPEAT = 20
SOURCE_OFFSETS = [0.4, 0.5, 0.6, 0.7, 0.8]
TARGET_OFFSET = 0.6


def sphere(x1: float, x2: float, b: float) -> float:
    return (x1 - b) ** 2 + (x2 - b) ** 2


def ellipsoid(x1: float, x2: float, b: float) -> float:
    scale = 5 ** 2
    return (x1 - b) ** 2 + scale * (x2 - b) ** 2


def rot_ellipsoid(x1: float, x2: float, b: float) -> float:
    rot_x1 = math.sqrt(3.0) / 2.0 * x1 + 1.0 / 2.0 * x2
    rot_x2 = 1.0 / 2.0 * x1 + math.sqrt(3.0) / 2.0 * x2
    return ellipsoid(rot_x1, rot_x2, b)


def evaluate(optimizer: CMA, f: Callable[[float, float, float], float]) -> float:
    f_best = None
    solutions = []
    n_evaluations = 50

    for _ in range(n_evaluations):
        x = optimizer.ask()
        value = f(x[0], x[1], TARGET_OFFSET)
        if f_best is None or f_best > value:
            f_best = value

        solutions.append((x, value))
        if len(solutions) == optimizer.population_size:
            optimizer.tell(solutions)
            solutions = []
    return f_best


def cmaes_best_sphere(seed: int, n_repeat: int) -> float:
    f_bests = []
    for i in range(n_repeat):
        # Employ N(0.5, 0.2^2) as a non-informative distribution.
        optimizer = CMA(
            mean=np.array([0.5, 0.5], dtype=float),
            sigma=0.2,
            population_size=8,
            seed=seed + i,
        )
        f_best = evaluate(
            optimizer,
            sphere,
        )
        f_bests.append(f_best)
    return np.mean(f_bests)


def ws_cmaes_best_sphere(source_offset: float, seed: int, n_repeat: int) -> float:
    rng = np.random.RandomState(seed)
    n_source_solutions = 10000
    f_bests = []

    for i in range(n_repeat):
        source_solutions = []
        # Random sampling (bounded in [0, 1]).
        for _ in range(n_source_solutions):
            x = rng.random(2)
            value = sphere(x[0], x[1], source_offset)
            source_solutions.append((x, value))

        ws_mean, ws_sigma, ws_cov = get_starting_point(
            source_solutions, gamma=0.1, alpha=0.1
        )
        ws_cmaes_optimizer = CMA(
            mean=ws_mean, sigma=ws_sigma, cov=ws_cov, population_size=8, seed=seed + i
        )
        f_bests.append(evaluate(ws_cmaes_optimizer, sphere))
    return np.mean(f_bests)


def similarity_sphere(seed: int, n_repeat: int) -> None:

    # CMA-ES
    cmaes_sphere_best = cmaes_best_sphere(seed, n_repeat)
    print(f"cmaes sphere best: {cmaes_sphere_best})")

    # WS-CMA-ES
    ws_cmaes_sphere_bests = []
    for source_offset in SOURCE_OFFSETS:
        sphere_best = ws_cmaes_best_sphere(source_offset, seed, n_repeat)
        print(f"ws_cmaes sphere best: {sphere_best} (source_offset={source_offset})")

        ws_cmaes_sphere_bests.append(sphere_best)


def main():
    similarity_sphere(seed=1, n_repeat=20)


if __name__ == "__main__":
    main()
