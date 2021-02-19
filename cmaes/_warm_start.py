import math
import numpy as np

from typing import Tuple, List


def get_warm_start_mgd(
    source_solutions: List[Tuple[np.ndarray, float]],
    gamma: float = 0.1,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Estimates a promising distribution of the source task, then
    returns a multivariate gaussian distribution (the mean vector
    and the covariance matrix) used for initialization of the CMA-ES.

    Args:
        source_solutions:
            List of solutions (parameter, value) on a source task.

        gamma:
            top-(gamma x 100)% solutions are selected from a set of solutions
            on a source task. (default: 0.1).

        alpha:
            prior parameter for the initial covariance matrix (default: 0.1).

    Returns:
        The tuple of mean vector, sigma, and covariance matrix.
    """
    # Paper: https://arxiv.org/abs/2012.06932
    assert 0 < gamma <= 1, "gamma should be in (0, 1]"

    if len(source_solutions) == 0:
        raise ValueError("solutions should contain one or more items.")

    # Select top-(gamma x 100)% solutions
    source_solutions = sorted(source_solutions, key=lambda t: t[1])
    gamma_n = math.floor(len(source_solutions) * gamma)
    assert gamma_n >= 1, "One or more solutions must be selected from a source task"
    dim = len(source_solutions[0][0])
    top_gamma_solutions = np.empty(
        shape=(
            gamma_n,
            dim,
        ),
        dtype=float,
    )
    for i in range(gamma_n):
        top_gamma_solutions[i] = source_solutions[i][0]

    # Estimation of a Promising Distribution of a Source Task.
    first_term = alpha ** 2 * np.eye(dim)
    cov_term = np.zeros(shape=(dim, dim), dtype=float)
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
        dtype=float,
    )
    for i in range(gamma_n):
        mean_term += top_gamma_solutions[i, :].reshape(dim, 1)
    mean_term /= gamma_n

    third_term = np.dot(mean_term, mean_term.T)
    mu = mean_term
    mean = mu[:, 0]
    Sigma = first_term + second_term - third_term
    det_sigma = np.linalg.det(Sigma)
    sigma = math.pow(det_sigma, 1.0 / 2.0 / dim)
    cov = Sigma / math.pow(det_sigma, 1.0 / dim)
    return mean, sigma, cov
