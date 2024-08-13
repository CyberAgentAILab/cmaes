from __future__ import annotations

import math
import numpy as np

from typing import Any
from typing import cast
from typing import Optional


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CatCMA:
    """CatCMA stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

            import numpy as np
            from cmaes import CatCMA

            def sphere_com(x, c):
                return sum(x*x) + len(c) - sum(c[:,0])

            optimizer = CatCMA(mean=3 * np.ones(3), sigma=2.0, cat_num=np.array([3, 3, 3]))

            for generation in range(50):
                solutions = []
                for _ in range(optimizer.population_size):
                    # Ask a parameter
                    x, c = optimizer.ask()
                    value = sphere_com(x, c)
                    solutions.append(((x, c), value))
                    print(f"#{generation} {value}")

                # Tell evaluation values.
                optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multivariate gaussian distribution.

        sigma:
            Initial standard deviation of covariance matrix.

        cat_num:
            Numbers of categories.

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

        cat_param:
            A parameter of categorical distribution (optional).

        margin:
            A margin (lower bound) of categorical distribution (optional).

        min_eigenvalue:
            Lower bound of eigenvalue of multivariate Gaussian distribution (optional).
    """

    # Paper: https://arxiv.org/abs/2405.09962

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        cat_num: np.ndarray,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
        cat_param: Optional[np.ndarray] = None,
        margin: Optional[np.ndarray] = None,
        min_eigenvalue: Optional[float] = None,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        self._n_co = len(mean)
        self._n_ca = len(cat_num)
        self._n = self._n_co + self._n_ca
        assert self._n_co > 1, "The dimension of mean must be larger than 1"
        assert self._n_ca > 0, "The dimension of categorical variable must be positive"
        assert np.all(cat_num > 1), "The number of categories must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(self._n))
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # CatCMA assumes that the weights of the lower half are zero.
        # (CMA uses negative weights while CatCMA uses positive weights.)
        weights_prime = np.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1) if i < mu else 0
                for i in range(population_size)
            ]
        )
        weights = weights_prime / weights_prime.sum()
        mu_eff = 1 / ((weights**2).sum())

        # learning rate for the rank-one update
        alpha_cov = 2
        c1 = alpha_cov / ((self._n_co + 1.3) ** 2 + mu_eff)
        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((self._n_co + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        cm = 1

        # learning rate for the cumulation for the step-size control
        c_sigma = (mu_eff + 2) / (self._n_co + mu_eff + 5)
        d_sigma = (
            1 + 2 * max(0, math.sqrt((mu_eff - 1) / (self._n_co + 1)) - 1) + c_sigma
        )
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update
        cc = (4 + mu_eff / self._n_co) / (self._n_co + 4 + 2 * mu_eff / self._n_co)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._popsize = population_size
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)||
        self._chi_n = math.sqrt(self._n_co) * (
            1.0 - (1.0 / (4.0 * self._n_co)) + 1.0 / (21.0 * (self._n_co**2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(self._n_co)
        self._pc = np.zeros(self._n_co)

        self._mean = mean.copy()

        if cov is None:
            self._C = np.eye(self._n_co)
        else:
            assert cov.shape == (
                self._n_co,
                self._n_co,
            ), "Invalid shape of covariance matrix"
            self._C = cov

        self._sigma = sigma
        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        # categorical distribution
        # Parameters in categorical distribution with fewer categories
        # must be zero-padded at the end.
        self._K = cat_num
        self._Kmax = np.max(self._K)
        if cat_param is None:
            self._q = np.zeros((self._n_ca, self._Kmax))
            for i in range(self._n_ca):
                self._q[i, : self._K[i]] = 1 / self._K[i]
        else:
            assert cat_param.shape == (
                self._n_ca,
                self._Kmax,
            ), "Invalid shape of categorical distribution parameter"
            for i in range(self._n_ca):
                assert np.all(cat_param[i, self._K[i] :] == 0), (
                    "Parameters in categorical distribution with fewer categories "
                    "must be zero-padded at the end"
                )
            assert np.all(
                (cat_param >= 0) & (cat_param <= 1)
            ), "All elements in categorical distribution parameter must be between 0 and 1"
            assert np.allclose(
                np.sum(cat_param, axis=1), 1
            ), "Each row in categorical distribution parameter must sum to 1"
            self._q = cat_param

        self._q_min = (
            margin
            if margin is not None
            else (1 - 0.73 ** (1 / self._n_ca)) / (self._K - 1)
        )
        self._min_eigenvalue = min_eigenvalue if min_eigenvalue is not None else 1e-30

        # ASNG
        self._param_sum = np.sum(cat_num - 1)
        self._alpha = 1.5
        self._delta_init = 1.0
        self._Delta = 1.0
        self._Delta_max = np.inf
        self._gamma = 0.0
        self._s = np.zeros(self._param_sum)
        self._delta = self._delta_init / self._Delta
        self._eps = self._delta

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0
        self._rng = np.random.RandomState(seed)

        # Termination criteria
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * self._n_co / population_size)
        self._funhist_values = np.empty(self._funhist_term)

    def __getstate__(self) -> dict[str, Any]:
        attrs = {}
        for name in self.__dict__:
            # Remove _rng in pickle serialized object.
            if name == "_rng":
                continue
            if name == "_C":
                sym1d = _compress_symmetric(self._C)
                attrs["_c_1d"] = sym1d
                continue
            attrs[name] = getattr(self, name)
        return attrs

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["_C"] = _decompress_symmetric(state["_c_1d"])
        del state["_c_1d"]
        self.__dict__.update(state)
        # Set _rng for unpickled object.
        setattr(self, "_rng", np.random.RandomState())

    @property
    def cont_dim(self) -> int:
        """A number of dimensions of continuous variable"""
        return self._n_co

    @property
    def cat_dim(self) -> int:
        """A number of dimensions of categorical variable"""
        return self._n_ca

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n

    @property
    def cat_num(self) -> np.ndarray:
        """Numbers of categories"""
        return self._K

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    @property
    def mean(self) -> np.ndarray:
        """Mean Vector"""
        return self._mean

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)

    def set_bounds(self, bounds: Optional[np.ndarray]) -> None:
        """Update boundary constraints"""
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "invalid bounds"
        self._bounds = bounds

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x, c = self._sample_solution()
            if self._is_feasible(x):
                return x, c
        x, c = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x, c

    def _eigen_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D**2)), B.T)

        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> tuple[np.ndarray, np.ndarray]:
        # x : continuous variable
        B, D = self._eigen_decomposition()
        z = self._rng.randn(self._n_co)  # ~ N(0, I)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        # c : categorical variable
        # Categorical variables are one-hot encoded.
        # Variables with fewer categories are zero-padded at the end.
        rand_q = self._rng.rand(self._n_ca, 1)
        cum_q = self._q.cumsum(axis=1)
        c = (cum_q - self._q <= rand_q) & (rand_q < cum_q)
        return x, c

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

    def tell(
        self, solutions: list[tuple[tuple[np.ndarray, np.ndarray], float]]
    ) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0][0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores best evaluation values of the
        # last 'self._funhist_term' generations.
        funhist_idx = self.generation % self._funhist_term
        self._funhist_values[funhist_idx] = solutions[0][1]

        # Sample new population of search_points, for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        x_k = np.array([s[0][0] for s in solutions])  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        C_2 = cast(
            np.ndarray, cast(np.ndarray, B.dot(np.diag(1 / D))).dot(B.T)
        )  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * C_2.dot(y_w)

        norm_p_sigma = np.linalg.norm(self._p_sigma)
        self._sigma *= np.exp(
            (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
        )
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(
            1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (self._n_co + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0

        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)
        assert delta_h_sigma <= 1

        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(self._weights, y_k)]), axis=0
        )
        self._C = (
            (
                1
                + self._c1 * delta_h_sigma
                - self._c1
                - self._cmu * np.sum(self._weights)
            )
            * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )

        # Post-processing to prevent the minimum eigenvalue from becoming too small
        B, D = self._eigen_decomposition()
        sigma_min = np.sqrt(self._min_eigenvalue / np.min(D))
        self._sigma = max(self._sigma, sigma_min)

        # Update of categorical distribution
        c = np.array([s[0][1] for s in solutions])
        ngrad = (self._weights[:, np.newaxis, np.newaxis] * (c - self._q)).sum(axis=0)

        # Approximation of the square root of the fisher information matrix :
        # Appendix B in https://proceedings.mlr.press/v97/akimoto19a.html
        sl = []
        for i, K in enumerate(self._K):
            q_i = self._q[i, : K - 1]
            q_i_K = self._q[i, K - 1]
            s_i = 1.0 / np.sqrt(q_i) * ngrad[i, : K - 1]
            s_i += np.sqrt(q_i) * ngrad[i, : K - 1].sum() / (q_i_K + np.sqrt(q_i_K))
            sl += list(s_i)
        ngrad_sqF = np.array(sl)

        pnorm = np.sqrt(np.dot(ngrad_sqF, ngrad_sqF)) + 1e-30
        self._eps = self._delta / pnorm
        self._q += self._eps * ngrad

        # Update of ASNG
        self._delta = self._delta_init / self._Delta
        beta = self._delta / (self._param_sum**0.5)
        self._s = (1 - beta) * self._s + np.sqrt(beta * (2 - beta)) * ngrad_sqF / pnorm
        self._gamma = (1 - beta) ** 2 * self._gamma + beta * (2 - beta)
        self._Delta *= np.exp(
            beta * (self._gamma - np.dot(self._s, self._s) / self._alpha)
        )
        self._Delta = min(self._Delta, self._Delta_max)

        # Margin Correction
        for i in range(self._n_ca):
            Ki = self._K[i]
            self._q[i, :Ki] = np.maximum(self._q[i, :Ki], self._q_min[i])
            q_sum = self._q[i, :Ki].sum()
            tmp = q_sum - self._q_min[i] * Ki
            self._q[i, :Ki] -= (q_sum - 1) * (self._q[i, :Ki] - self._q_min[i]) / tmp
            self._q[i, :Ki] /= self._q[i, :Ki].sum()

    def should_stop(self) -> bool:
        B, D = self._eigen_decomposition()

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
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


def _compress_symmetric(sym2d: np.ndarray) -> np.ndarray:
    assert len(sym2d.shape) == 2 and sym2d.shape[0] == sym2d.shape[1]
    n = sym2d.shape[0]
    dim = (n * (n + 1)) // 2
    sym1d = np.zeros(dim)
    start = 0
    for i in range(n):
        sym1d[start : start + n - i] = sym2d[i][i:]  # noqa: E203
        start += n - i
    return sym1d


def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray:
    n = int(np.sqrt(sym1d.size * 2))
    assert (n * (n + 1)) // 2 == sym1d.size
    R, C = np.triu_indices(n)
    out = np.zeros((n, n), dtype=sym1d.dtype)
    out[R, C] = sym1d
    out[C, R] = sym1d
    return out
