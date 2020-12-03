import math
import numpy as np

from typing import Tuple, List


def get_sigma_and_mean(
    solutions: List[Tuple[np.ndarray, float]],
    ratio_high_rank_gamma: float = 0.1,
    normal_prior_alpha: float = 0.1,
) -> Tuple[np.ndarray, float, np.ndarray]:
    if len(solutions) == 0:
        raise ValueError("solutions should contain one or more items.")

    solutions = sorted(solutions, key=lambda t: t[1])
    gamma_n = int(len(solutions) * ratio_high_rank_gamma)
    dim = len(solutions[0])

    top_gamma_solutions = np.empty(
        shape=(
            gamma_n,
            dim,
        ),
        dtype=np.float,
    )
    for i in range(gamma_n):
        top_gamma_solutions[i] = solutions[i]

    first_term = normal_prior_alpha ** 2 * np.eye(dim)
    cov_term = np.zeros(shape=(dim, dim), dtype=np.float)
    for i in range(gamma_n):
        cov_term += np.dot(
            top_gamma_solutions[i, :].reshape(dim, 1),
            top_gamma_solutions[i, :].reshape(dim, 1).T,
        )

    second_term = cov_term / gamma_n
    mean_term = np.zeros(
        shape=(
            dim,
            1,
        ),
        dtype=np.float,
    )
    for i in range(gamma_n):
        mean_term += top_gamma_solutions[i, :].reshape(dim, 1)
    mean_term /= gamma_n

    third_term = np.dot(mean_term, mean_term.T)
    mu = mean_term
    mean = mu[:, 0]
    det_sigma = np.linalg.det(first_term + second_term - third_term)
    sigma = math.pow(det_sigma, 1.0 / 2.0 / dim)
    cov = sigma / math.pow(det_sigma, 1.0 / dim)
    return mean, sigma, cov
