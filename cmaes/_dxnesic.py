from __future__ import annotations

import math
import sys

import numpy as np

from typing import cast
from typing import Optional


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class DXNESIC:
    """DX-NES-IC stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

           import numpy as np
           from cmaes import DXNESIC

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = DXNESIC(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).
    """

    # Paper: https://ieeexplore.ieee.org/abstract/document/9504865

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))
        assert population_size > 0, "popsize must be non-zero positive value."

        w_rank_hat = np.log(population_size / 2 + 1) - np.log(
            np.arange(1, population_size + 1)
        )
        w_rank_hat[np.where(w_rank_hat < 0)] = 0
        w_rank = w_rank_hat / sum(w_rank_hat) - (1.0 / population_size)
        mu_eff = 1 / sum((w_rank + (1.0 / population_size)) ** 2)

        # learning rate for the cumulation for the step-size control
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # distance weight parameter
        h_inv = _get_h_inv(n_dim)

        self._n_dim = n_dim
        self._popsize = population_size
        self._mu_eff = mu_eff

        self._h_inv = h_inv
        self._c_sigma = c_sigma

        # E||N(0, I)||
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim**2))
        )

        # weights
        self._w_rank = w_rank
        self._w_rank_hat = w_rank_hat

        # for antithetic sampling
        self._zsym: Optional[np.ndarray] = None

        # learning rate
        self._eta_mean = 1.0
        self._eta_move_sigma = 1.0
        self._c_gamma = 1.0 / (3.0 * (n_dim - 1.0))
        self._d_gamma = min(1.0, n_dim / population_size)
        self._gamma = 1.0

        # evolution path
        self._p_sigma = np.zeros(n_dim)

        # distribution parameter
        self._mean = mean.copy()
        self._sigma = sigma
        self._B = np.eye(n_dim)

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0
        self._rng = np.random.RandomState(seed)

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    def _alpha_dist(self, num_feasible: int) -> float:
        return (
            self._h_inv
            * min(1.0, math.sqrt(float(self._popsize) / self._n_dim))
            * math.sqrt(float(num_feasible) / self._popsize)
        )

    def _w_dist_hat(self, z: np.ndarray, num_feasible: int) -> float:
        return math.exp(self._alpha_dist(num_feasible) * np.linalg.norm(z))

    def _eta_stag_sigma(self, num_feasible: int) -> float:
        return math.tanh(
            (0.024 * num_feasible + 0.7 * self._n_dim + 20.0) / (self._n_dim + 12.0)
        )

    def _eta_conv_sigma(self, num_feasible: int) -> float:
        return 2.0 * math.tanh(
            (0.025 * num_feasible + 0.75 * self._n_dim + 10.0) / (self._n_dim + 4.0)
        )

    def _eta_move_B(self, num_feasible: int) -> float:
        return (
            180
            * self._n_dim
            * math.tanh(0.02 * num_feasible)
            / (47 * (self._n_dim**2) + 6400)
        )

    def _eta_stag_B(self, num_feasible: int) -> float:
        return (
            168
            * self._n_dim
            * math.tanh(0.02 * num_feasible)
            / (47 * (self._n_dim**2) + 6400)
        )

    def _eta_conv_B(self, num_feasible: int) -> float:
        return (
            12
            * self._n_dim
            * math.tanh(0.02 * num_feasible)
            / (47 * (self._n_dim**2) + 6400)
        )

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)

    def set_bounds(self, bounds: Optional[np.ndarray]) -> None:
        """Update boundary constraints"""
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "invalid bounds"
        self._bounds = bounds

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _sample_solution(self) -> np.ndarray:
        # antithetic sampling
        if self._zsym is None:
            z = self._rng.randn(self._n_dim)  # ~ N(0, I)
            self._zsym = z
        else:
            z = -self._zsym
            self._zsym = None
        x = self._mean + self._sigma * self._B.dot(z)  # ~ N(m, σ^2 B B^T)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(
            bool,
            np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        # counting # feasible solutions
        lamb_feas = len([s[1] for s in solutions if s[1] < sys.maxsize])

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        z_k = np.array(
            [
                np.linalg.inv(self._sigma * self._B).dot(s[0] - self._mean)
                for s in solutions
            ]
        )

        # Evolution path
        z_w = np.sum(z_k.T * self._w_rank, axis=1)
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * z_w

        norm_p_sigma = np.linalg.norm(self._p_sigma)

        # switching learning rate depending on search situation
        movement_phase = norm_p_sigma >= self._chi_n

        # distance weight
        w_dist_tmp = np.array(
            [
                self._w_rank_hat[i] * self._w_dist_hat(z_k[i, :], lamb_feas)
                for i in range(self.population_size)
            ]
        )
        w_dist = w_dist_tmp / sum(w_dist_tmp) - 1.0 / self.population_size

        # switching weights and learning rate
        w = w_dist if movement_phase else self._w_rank
        eta_sigma = (
            self._eta_move_sigma
            if norm_p_sigma >= self._chi_n
            else (
                self._eta_stag_sigma(lamb_feas)
                if norm_p_sigma >= 0.1 * self._chi_n
                else self._eta_conv_sigma(lamb_feas)
            )
        )
        eta_B = (
            self._eta_move_B(lamb_feas)
            if norm_p_sigma >= self._chi_n
            else (
                self._eta_stag_B(lamb_feas)
                if norm_p_sigma >= 0.1 * self._chi_n
                else self._eta_conv_B(lamb_feas)
            )
        )

        # natural gradient estimation in local coordinate
        G_delta = np.sum(
            [w[i] * z_k[i, :] for i in range(self.population_size)], axis=0
        )
        G_M = np.sum(
            [
                w[i] * (np.outer(z_k[i, :], z_k[i, :]) - np.eye(self._n_dim))
                for i in range(self.population_size)
            ],
            axis=0,
        )
        G_sigma = G_M.trace() / self._n_dim
        G_B = G_M - G_sigma * np.eye(self._n_dim)

        # parameter update
        bBBT = self._B @ self._B.T
        self._mean += self._eta_mean * self._sigma * np.dot(self._B, G_delta)
        self._sigma *= math.exp((eta_sigma / 2.0) * G_sigma)
        # self._B = self._B.dot(expm((eta_B / 2.0) * G_B))
        self._B = self._B.dot(_expm((eta_B / 2.0) * G_B))
        aBBT = self._B @ self._B.T

        # emphasizing expansion
        e, v = np.linalg.eigh(bBBT)
        tau_vec = [
            (v[:, i].reshape(self._n_dim, 1).T @ aBBT @ v[:, i].reshape(self._n_dim, 1))
            / (
                v[:, i].reshape(self._n_dim, 1).T
                @ bBBT
                @ v[:, i].reshape(self._n_dim, 1)
            )
            - 1
            for i in range(self._n_dim)
        ]
        flg_tau = [1.0 if tau_vec[i] > 0 else 0.0 for i in range(self._n_dim)]
        tau = max(tau_vec)
        gamma = max(
            (1.0 - self._c_gamma) * self._gamma
            + self._c_gamma * math.sqrt(1.0 + self._d_gamma * tau),
            1.0,
        )
        if movement_phase:
            Q = (gamma - 1.0) * np.sum(
                [flg_tau[i] * np.outer(v[:, i], v[:, i]) for i in range(self._n_dim)],
                axis=0,
            ) + np.eye(self._n_dim)
            stepQ = math.pow(np.linalg.det(Q), 1.0 / self._n_dim)
            self._sigma *= stepQ
            self._B = Q @ self._B / stepQ

    def should_stop(self) -> bool:
        A = self._B.dot(self._B.T)
        A = (A + A.T) / 2
        E2, V = np.linalg.eigh(A)
        E = np.sqrt(np.where(E2 < 0, _EPS, E2))
        diagA = np.diag(A)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(E) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(diagA))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * E[i] * V[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(E) / np.min(E)
        if condition_cov > self._tolconditioncov:
            return True

        return False


def _is_valid_bounds(bounds: Optional[np.ndarray], mean: np.ndarray) -> bool:
    if bounds is None:
        return True
    if (mean.size, 2) != bounds.shape:
        return False
    if not np.all(bounds[:, 0] <= mean):
        return False
    if not np.all(mean <= bounds[:, 1]):
        return False
    return True


def _get_h_inv(dim: int) -> float:
    def f(a: float) -> float:
        return ((1.0 + a * a) * math.exp(a * a / 2.0) / 0.24) - 10.0 - dim

    def f_prime(a: float) -> float:
        return (1.0 / 0.24) * a * math.exp(a * a / 2.0) * (3.0 + a * a)

    h_inv = 6.0
    while abs(f(h_inv)) > 1e-10:
        last = h_inv
        h_inv = h_inv - 0.5 * (f(h_inv) / f_prime(h_inv))
        if abs(h_inv - last) < 1e-16:
            # Exit early since no further improvements are happening
            break
    return h_inv


def _expm(mat: np.ndarray) -> np.ndarray:
    D, U = np.linalg.eigh(mat)
    expD = np.exp(D)
    return U @ np.diag(expD) @ U.T
