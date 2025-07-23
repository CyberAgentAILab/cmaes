from __future__ import annotations

import math
import numpy as np

from typing import Any
from typing import cast
from typing import Optional

from scipy.spatial import Voronoi
from scipy.stats import chi2
from scipy.stats import norm

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CMASoP:
    """CMA-ES-SoP stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::
            import numpy as np
            from cmaes.cma_sop import CMASoP

            # numbers of dimensions in each subspace
            subspace_dim_list = [2, 3, 5]
            cont_dim = 10

            # numbers of points in each subspace
            point_num_list = [10, 20, 40]

            # number of total dimensions
            dim = int(np.sum(subspace_dim_list) + cont_dim)

            # objective function
            def quadratic(x):
                coef = 1000 ** (np.arange(dim) / float(dim - 1))
                return np.sum(x ** 2)

            # sets_of_points (on [-5, 5])
            subspace_num = len(subspace_dim_list)
            sets_of_points = [(
                2 * np.random.rand(point_num_list[i], subspace_dim_list[i]) - 1) * 5
            for i in range(subspace_num)]

            # the optimal solution is contained
            for i in range(subspace_num):
                sets_of_points[i][-1] = np.zeros(subspace_dim_list[i])
                np.random.shuffle(sets_of_points[i])

            # optimizer (CMA-ES-SoP)
            optimizer = CMASoP(
                sets_of_points=sets_of_points,
                mean=np.random.rand(dim) * 4 + 1,
                sigma=2.0,
            )

            best_eval = np.inf
            eval_count = 0

            for generation in range(400):
                solutions = []
                for _ in range(optimizer.population_size):
                    # Ask a parameter
                    x, enc_x = optimizer.ask()
                    value = quadratic(enc_x)

                    # save best eval
                    best_eval = np.min((best_eval, value))
                    eval_count += 1

                    solutions.append((x, value))

                # Tell evaluation values.
                optimizer.tell(solutions)

                print(f"#{generation} ({best_eval} {eval_count})")

                if best_eval < 1e-4 or optimizer.should_stop():
                    break


    Args:
        sets_of_points:
            List of points for each subspace.

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

        margin:
            A margin parameter (optional).

    """

    # Paper: https://arxiv.org/abs/2408.13046

    def __init__(
        self,
        sets_of_points: np.ndarray,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
        margin: Optional[float] = None,
    ):
        # same initialization procedure as for naive cma
        self._naive_cma_init_(
            mean,
            sigma,
            bounds,
            n_max_resampling,
            seed,
            population_size,
            cov,
        )

        # preprocess of sets of points
        if sets_of_points is not None:
            self._sets_of_points = sets_of_points
            self._zd = [ds.shape[1] for ds in sets_of_points]
            self._point_num = [ds.shape[0] for ds in sets_of_points]
            self._vor_list = [Voronoi(ds) for ds in sets_of_points]
            self._subspace_mask = None
            self._neighbor_matrices = self._get_neighbor_matrices()
        else:
            self._zd = []

        # setting for margin correction and adaptation
        self._margin_target = (
            margin if margin is not None else 1 / (self._n_dim * self._popsize)
        )
        self._margin = self._margin_target * np.ones_like(self._zd)
        self._margin_coeff = 1 + 1 / self._n_dim if self._margin_target > 0 else 0

    def _naive_cma_init_(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
    ) -> None:
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        assert n_dim > 0, "The dimension of mean must be positive"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        weights_prime = np.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1)
                for i in range(population_size)
            ]
        )
        weights_prime[weights_prime < 0] = 0
        weights = weights_prime / weights_prime.sum()
        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)

        # learning rate for the rank-one update
        alpha_cov = 2
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)

        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        cm = 1

        # learning rate for the cumulation for the step-size control
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
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
        self._chi_n = math.sqrt(self._n_dim) * (
            1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim**2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._pc = np.zeros(n_dim)

        self._mean = mean.copy()

        if cov is None:
            self._C = np.eye(n_dim)
        else:
            assert cov.shape == (n_dim, n_dim), "Invalid shape of covariance matrix"
            self._C = cov

        self._sigma = sigma
        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

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

    def _get_neighbor_matrices(self) -> list:
        try:
            # if already computed
            return self._neighbor_matrices
        except AttributeError:

            def neighbor_matrix(i: int) -> np.ndarray:
                point_num = self._point_num[i]
                ridge_points = self._vor_list[i].ridge_points
                res = np.zeros((point_num, point_num), dtype=bool)
                res[ridge_points[:, 0], ridge_points[:, 1]] = True
                return res | res.T

            # compute neighboring points
            self._neighbor_matrices = [
                neighbor_matrix(i) for i in range(len(self._sets_of_points))
            ]
            return self._neighbor_matrices

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
            x = self._sample_solution()
            if self._is_feasible(x):
                enc_x = self._encoding(x)  # eoncoded solution
                return x, enc_x

        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        enc_x = self._encoding(x)  # eoncoded solution
        return x, enc_x

    def _eigen_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D**2)), B.T)

        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
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

    def _encoding(self, X: np.ndarray) -> np.ndarray:
        X_ndim = X.ndim
        X = np.atleast_2d(X)

        num_cont = self._n_dim - np.sum(self._zd)  # = N_continuous
        if num_cont == self._n_dim:
            return X

        # encoding
        closest_idx = self._get_closest_point_index(X)
        X_z_enc = np.hstack(
            [self._sets_of_points[i][closest_idx[i]] for i in range(len(self._zd))]
        )
        if X_ndim == 1:
            return np.hstack((X[:, :num_cont], X_z_enc))[0]
        else:
            return np.hstack((X[:, :num_cont], X_z_enc))

    def _get_closest_point_index(self, X: np.ndarray) -> list[Any]:
        X = np.atleast_2d(X)

        # return the closest point in i-th subspace
        def get_closest(i: int) -> np.ndarray:
            X_z = X[:, self._get_subspace_mask()[i]]
            vor = self._vor_list[i]
            dist2 = ((X_z[:, None, :] - vor.points[None, :, :]) ** 2).sum(axis=2)
            return np.argmin(dist2, axis=1)

        return [get_closest(i) for i in range(len(self._zd))]

    def _get_subspace_mask(self) -> np.ndarray:
        if self._subspace_mask is not None:
            return self._subspace_mask
        else:
            self._subspace_mask = np.zeros((len(self._zd), self._n_dim), dtype=bool)
            cont_dim = self._n_dim - np.sum(self._zd)
            subspace_range = np.concatenate(
                [[cont_dim], cont_dim + np.cumsum(self._zd)]
            )

            for i in range(len(self._zd)):
                self._subspace_mask[i, subspace_range[i] : subspace_range[i + 1]] = True

            return self._subspace_mask

    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""
        self._naive_cma_update(solutions)

        # margin correction (if self.margin = 0, this behaves as CMA-ES)
        if np.sum(self._zd) > 0 and self._margin_target > 0:
            self._margin_correction()

    def _naive_cma_update(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        # Sample new population of search_points, for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
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
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
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

    def _get_neighbor_indexes(self, m: np.ndarray) -> list[Any]:
        # get neiboring points to given point
        closest_index = np.array(self._get_closest_point_index(m))[:, 0]
        return [
            self._get_neighbor_matrices()[i][closest_index[i]]
            for i in range(len(self._zd))
        ]

    def _margin_correction(self) -> None:
        nearest_indexes = self._get_neighbor_indexes(self._mean)

        for i in range(len(self._zd)):
            # margin correction (eq. (10)-(15))
            CI = np.sqrt(chi2.ppf(q=1.0 - self._margin[i], df=1))
            target_nearest_points = self._sets_of_points[i][nearest_indexes[i]]
            m_z = self._mean[self._get_subspace_mask()[i]]

            if len(target_nearest_points) == 0:
                return

            self._rng.shuffle(target_nearest_points)

            for x_near_z in target_nearest_points:

                y_near_z = (x_near_z - m_z) / self._sigma
                y_near = np.zeros(self._n_dim)
                y_near[self._get_subspace_mask()[i]] = y_near_z  # eq. (14)

                B, D = self._eigen_decomposition()
                invSqrtC = B @ np.diag(1 / D) @ B.T

                z_near = np.dot(invSqrtC, y_near)
                dist = np.linalg.norm(z_near) / 2  # midpoint (eq. (13))

                if dist > CI:
                    beta = (dist**2 - CI**2) / ((dist**2) * (CI**2))
                    self._C = self._C + beta * np.outer(y_near, y_near)
                    self._B, self._D = None, None

            # margin adaptation (eq. (16))
            Y_near_z = (target_nearest_points - m_z) / self._sigma
            Y_near = np.zeros((len(Y_near_z), self._n_dim))
            Y_near[:, self._get_subspace_mask()[i]] = Y_near_z

            B, D = self._eigen_decomposition()
            corrected_invSqrtC = B @ np.diag(1 / D) @ B.T
            self._B, self._D = None, None

            Z_near = np.dot(Y_near, corrected_invSqrtC)
            dist = np.linalg.norm(Z_near, axis=1) / 2  # midpoint

            current_margin = np.mean(1 - norm.cdf(dist))

            # eq. (16)
            if current_margin > self._margin_target:
                self._margin[i] /= self._margin_coeff
            else:
                self._margin[i] *= self._margin_coeff

    def should_stop(self) -> bool:
        B, D = self._eigen_decomposition()
        dC = np.diag(self._C)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        if np.all(self._sigma * dC < self._tolx) and np.all(
            self._sigma * self._pc < self._tolx
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(D) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(dC))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
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
