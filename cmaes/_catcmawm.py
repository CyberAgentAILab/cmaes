from __future__ import annotations

import functools
import numpy as np
import math

from dataclasses import dataclass, field

from typing import cast
from typing import List, Sequence, Union, Tuple, Optional

import warnings

try:
    from scipy import stats

    chi2_ppf = functools.partial(stats.chi2.ppf, df=1)
    norm_cdf = stats.norm.cdf
except ImportError:
    from cmaes._stats import chi2_ppf  # type: ignore
    from cmaes._stats import norm_cdf


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class CatCMAwM:
    """CatCMA with Margin stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

            import numpy as np
            from cmaes import CatCMAwM

            def SphereIntCOM(x, z, c):
                return sum(x*x) + sum(z*z) + len(c) - sum(c[:,0])

            X = [[-3.0, 3.0], [-4.0, 4.0]]
            Z = [[-1, 0, 1], [-2, -1, 0, 1, 2]]
            C = [5, 6]

            optimizer = CatCMAwM(x_space=X, z_space=Z, c_space=C)

            for generation in range(50):
                solutions = []
                for _ in range(optimizer.population_size):
                    # Ask a parameter
                    sol = optimizer.ask()
                    value = SphereIntCOM(sol.x, sol.z, sol.c)
                    print(f"#{generation} {value}")
                    solutions.append((sol, value))

                # Tell evaluation values
                optimizer.tell(solutions)

    Args:

        x_space:
            The search space for continuous variables.
            Provide as a 2-dimensional sequence (e.g., a list of lists),
            where each row is [lower_bound, upper_bound] for a continuous variable.
            If there are no continuous variables, this parameter can be omitted.
            Example: [[-3.0, 3.0], [0.0, 5.0], [-np.inf, np.inf]]

        z_space:
            The set of possible values for each integer variable.
            Provide as a list of lists, where each inner list contains the valid
            (sorted) integer or discretized values for that variable.
            If there are no integer variables, this parameter can be omitted.
            Example: [[-2, -1, 0, 1, 2], [0.01, 0.1, 1]]
            Note: For binary variables (i.e., variables that can only take two distinct values),
            it is generally recommended to use the categorical variable representation via
            `c_space` rather than treating them as integer variables.

        c_space:
            The shape of the categorical variables' domain.
            Provide as a 1-dimensional sequence (e.g., a list)
            where each element specifies the number of categories (integer > 1)
            for each categorical variable.
            If there are no categorical variables, this parameter can be omitted.
            Example: [3, 3, 2, 10]
            Note: Binary variables (with only two possible values) should be represented as
            categorical variables here, rather than as integer variables in `z_space`.

        population_size:
            A population size (optional).

        mean:
            Initial mean vector of multivariate gaussian distribution (optional).

        sigma:
            Initial step-size of multivariate gaussian distribution (optional).

        cov:
            Initial covariance matrix of multivariate gaussian distribution (optional).

        cat_param:
            Initial parameter of categorical distribution (optional).

        seed:
            A seed number (optional).
    """

    # Paper: https://arxiv.org/abs/2504.07884

    @dataclass(frozen=True)
    class Solution:
        x: Optional[np.ndarray] = None  # continuous variable
        z: Optional[np.ndarray] = None  # integer variable
        c: Optional[np.ndarray] = None  # categorical variable
        _v_raw: Optional[np.ndarray] = field(default=None, repr=False)  # internal use

    def __init__(
        self,
        x_space: Optional[Sequence[Sequence[float]]] = None,
        z_space: Optional[Sequence[Sequence[Union[int, float]]]] = None,
        c_space: Optional[Sequence[int]] = None,
        population_size: Optional[int] = None,
        mean: Optional[np.ndarray] = None,
        sigma: Optional[float] = None,
        cov: Optional[np.ndarray] = None,
        cat_param: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        # Determine space sizes
        self._Nco = len(x_space) if x_space is not None else 0
        self._Nin = len(z_space) if z_space is not None else 0
        self._Nca = len(c_space) if c_space is not None else 0
        self._Nmi = self._Nco + self._Nin

        if self._Nmi + self._Nca <= 0:
            raise ValueError("The total number of dimensions must be positive.")

        self._use_continuous = self._Nco > 0
        self._use_integer = self._Nin > 0
        self._use_gaussian = self._Nmi > 0
        self._use_categorical = self._Nca > 0

        self._continuous_idx = np.arange(self._Nco)
        self._discrete_idx = np.arange(self._Nco, self._Nmi)

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(self._Nmi + self._Nca))
        if population_size <= 0:
            raise ValueError("population_size must be non-zero positive value.")
        self._popsize = population_size

        # --- CMA-ES weight (active covariance matrix adaptation) ---
        self._mu = self._popsize // 2
        weights_prime = np.array(
            [
                math.log((self._popsize + 1) / 2) - math.log(i + 1)
                for i in range(self._popsize)
            ]
        )
        self._mu_eff = (np.sum(weights_prime[: self._mu]) ** 2) / np.sum(
            weights_prime[: self._mu] ** 2
        )
        mu_eff_minus = (np.sum(weights_prime[self._mu :]) ** 2) / np.sum(
            weights_prime[self._mu :] ** 2
        )

        # learning rate for the rank-one update
        alpha_cov = 2
        self._c1 = alpha_cov / ((self._Nmi + 1.3) ** 2 + self._mu_eff)

        # learning rate for the rank-μ update
        self._cmu = min(
            1 - self._c1 - _EPS,  # _EPS is for large popsize.
            alpha_cov
            * (self._mu_eff - 2 + 1 / self._mu_eff)
            / ((self._Nmi + 2) ** 2 + alpha_cov * self._mu_eff / 2),
        )
        assert (
            self._c1 <= 1 - self._cmu
        ), "Invalid learning rate for the rank-one update."
        assert self._cmu <= 1 - self._c1, "Invalid learning rate for the rank-μ update."

        min_alpha = (
            0
            if self._Nmi == 0
            else min(
                1 + self._c1 / self._cmu,
                1 + (2 * mu_eff_minus) / (self._mu_eff + 2),
                (1 - self._c1 - self._cmu) / (self._Nmi * self._cmu),
            )
        )

        # TODO: Handle ranking ties when computing weights.
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        self._weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )

        # generation number
        self._g = 0

        self._rng = np.random.RandomState(seed)

        # --- initialization for each domain ---
        if self._use_integer:
            self._init_discretization(z_space)

        if self._use_gaussian:
            self._init_gaussian(x_space, mean, sigma, cov)

        if self._use_categorical:
            self._init_categorical(c_space, cat_param)

    def _init_discretization(
        self,
        z_space: Optional[Sequence[Sequence[Union[int, float]]]],
    ) -> None:
        assert z_space is not None, "z_space must not be None for integer variables."

        for i, row in enumerate(z_space):
            if len(row) < 2:
                raise ValueError(
                    f"z_space must be a sequence of arrays with length >= 2. "
                    f"Found length {len(row)} at index {i}: {row}"
                )
            if len(set(row)) < len(row):
                raise ValueError(
                    f"Elements in each array of z_space must be unique. "
                    f"Found duplicate at index {i}: {row}"
                )

        # Pad the row with its maximum value to reach the maximum row length
        max_length = max(len(row) for row in z_space)
        self._z_space = np.array(
            [
                np.pad(
                    np.array(sr),
                    pad_width=(0, max_length - len(sr)),
                    mode="constant",
                    constant_values=(sr[-1]),
                )
                for row in z_space
                for sr in [sorted(row)]
            ]
        )

        # discretization thresholds
        self._z_lim = (self._z_space[:, 1:] + self._z_space[:, :-1]) / 2

        # margin value for integer variables
        self._alpha = 1 - 0.73 ** (1 / (self._Nin + self._Nca))

        # mutation rates for integer variables
        self._pmut = (0.5 - _EPS) * np.ones(self._Nin)

        # successful integer mutation
        self._int_succ = np.zeros(self._Nin, dtype=bool)

    def _init_gaussian(
        self,
        x_space: Optional[Sequence[Sequence[float]]],
        mean: Optional[np.ndarray],
        sigma: Optional[float],
        cov: Optional[np.ndarray],
    ) -> None:
        if x_space is not None:
            self._x_space = np.asarray(x_space, dtype=float)
            if self._x_space.ndim != 2 or self._x_space.shape[1] != 2:
                raise ValueError(
                    f"x_space must be a two-dimensional array with shape (n, 2), "
                    f"but got shape {self._x_space.shape}."
                )
            invalid = np.where(self._x_space[:, 0] >= self._x_space[:, 1])[0]
            if invalid.size > 0:
                i = invalid[0]
                lb, ub = self._x_space[i, 0], self._x_space[i, 1]
                raise ValueError(
                    f"Lower bound must be less than upper bound at index {i}: {lb} >= {ub}"
                )

        # bounds for the mixed continuous and integer space
        if self._use_continuous and self._use_integer:
            lower_x = self._x_space[:, 0]
            upper_x = self._x_space[:, 1]
            lower_z = np.min(self._z_space, axis=1)
            upper_z = np.max(self._z_space, axis=1)
            lower_g = np.concatenate([lower_x, lower_z])
            upper_g = np.concatenate([upper_x, upper_z])

        # bounds for the integer space
        if not self._use_continuous and self._use_integer:
            lower_g = np.min(self._z_space, axis=1)
            upper_g = np.max(self._z_space, axis=1)

        # bounds for the continuous space
        if self._use_continuous and not self._use_integer:
            lower_g = self._x_space[:, 0]
            upper_g = self._x_space[:, 1]

        if mean is None:
            # Set initial mean to the center of the search space
            self._mean = np.zeros(self._Nmi)
            self._mean[(lower_g != -np.inf) & (upper_g != np.inf)] = (
                lower_g[(lower_g != -np.inf) & (upper_g != np.inf)]
                + upper_g[(lower_g != -np.inf) & (upper_g != np.inf)]
            ) / 2
            self._mean[(lower_g == -np.inf) & (upper_g != np.inf)] = (
                upper_g[(lower_g == -np.inf) & (upper_g != np.inf)] - 1
            )
            self._mean[(lower_g != -np.inf) & (upper_g == np.inf)] = (
                lower_g[(lower_g != -np.inf) & (upper_g == np.inf)] + 1
            )
        else:
            if len(mean) != self._Nmi:
                raise ValueError(
                    f"Invalid shape of mean: expected length {self._Nmi}, "
                    f"but got {len(mean)}."
                )
            self._mean = mean

        assert np.all(
            np.abs(self._mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}."

        if sigma is None:
            self._sigma = 1.0
        else:
            if sigma <= 0:
                raise ValueError("sigma must be non-zero positive value.")
            self._sigma = sigma

        if cov is None:
            # Set initial standard deviation to
            # width / 6 (continuous)
            # width / 5 (integer)
            width = np.minimum(self._mean - lower_g, upper_g - self._mean)
            width /= np.where(np.arange(self._Nmi) < self._Nco, 6, 5)
            self._C = np.diag(np.where(np.isfinite(width), width**2, 1.0))
        else:
            if cov.shape != (self._Nmi, self._Nmi):
                raise ValueError(
                    f"Invalid shape of covariance matrix: expected "
                    f"({self._Nmi}, {self._Nmi}), but got {cov.shape}."
                )
            self._C = cov

        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        # --- Other CMA-ES parameters ---
        # learning rate for the mean
        self._cm = 1.0

        # learning rate for the cumulation for the step-size control
        self._c_sigma = (self._mu_eff + 2) / (self._Nmi + self._mu_eff + 5)
        self._d_sigma = (
            1
            + 2 * max(0, math.sqrt((self._mu_eff - 1) / (self._Nmi + 1)) - 1)
            + self._c_sigma
        )
        assert (
            self._c_sigma < 1
        ), "Invalid learning rate for cumulation for the step-size control."

        # learning rate for cumulation for the rank-one update
        self._cc = (4 + self._mu_eff / self._Nmi) / (
            self._Nmi + 4 + 2 * self._mu_eff / self._Nmi
        )
        assert (
            self._cc <= 1
        ), "Invalid learning rate for cumulation for the rank-one update."

        # E||N(0, I_Nmi)||
        self._chi_n = math.sqrt(self._Nmi) * (
            1.0 - (1.0 / (4.0 * self._Nmi)) + 1.0 / (21.0 * (self._Nmi**2))
        )

        # evolution paths
        self._p_sigma = np.zeros(self._Nmi)
        self._pc = np.zeros(self._Nmi)

        # matrix for margin correction
        self._A = np.full(self._Nmi, 1.0)

        # minimum eigenvalue of covariance matrix
        self._min_eigenvalue = 1e-30

        # history of interquartile range of the unpenalized objective function values
        self._iqhist_term = 20 + math.ceil(3 * self._Nmi / self._popsize)
        self._iqhist_values: List[float] = []

        # termination criteria based on CMA-ES
        self._tolx = 1e-12 * self._sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14
        self._funhist_term = 10 + math.ceil(30 * self._Nmi / self._popsize)
        self._funhist_values = np.empty(self._funhist_term * 2)

    def _init_categorical(
        self,
        c_space: Optional[Sequence[int]],
        cat_param: Optional[np.ndarray],
    ) -> None:
        assert (
            c_space is not None
        ), "c_space must not be None for categorical variables."

        self._K = np.asarray(c_space, dtype=int)
        if not np.all(self._K >= 2):
            invalid = np.where(self._K < 2)[0][0]
            raise ValueError(
                f"All elements of c_space must be >= 2. "
                f"Found {self._K[invalid]} at index {invalid}."
            )

        self._Kmax = np.max(self._K)
        if cat_param is None:
            self._q = np.zeros((self._Nca, self._Kmax))
            for i in range(self._Nca):
                self._q[i, : self._K[i]] = 1 / self._K[i]
        else:
            if cat_param.shape != (self._Nca, self._Kmax):
                raise ValueError(
                    f"Invalid shape of categorical distribution parameter: "
                    f"expected ({self._Nca}, {self._Kmax}), got {cat_param.shape}."
                )
            for i in range(self._Nca):
                if not np.all(cat_param[i, self._K[i] :] == 0):
                    raise ValueError(
                        f"Parameters in categorical distribution with fewer categories "
                        f"must be zero-padded at the end. "
                        f"Non-zero padding found at row {i}: {cat_param[i]}"
                    )
            if not np.all((cat_param >= 0) & (cat_param <= 1)):
                raise ValueError(
                    "All elements in categorical distribution parameter "
                    "must be between 0 and 1."
                )
            if not np.allclose(np.sum(cat_param, axis=1), 1):
                raise ValueError(
                    "Each row in categorical distribution parameter must sum to 1."
                )
            self._q = cat_param

        # margin value for categorical variables
        self._qmin = (1 - 0.73 ** (1 / (self._Nin + self._Nca))) / (self._K - 1)

        # --- ASNG parameters ---
        # Adaptive Stochastic Natural Gradient method:
        # https://proceedings.mlr.press/v97/akimoto19a.html
        self._param_sum = np.sum(self._K - 1)
        self._alpha_snr = 1.5
        self._delta_init = 1.0
        self._Delta = 1.0
        self._Delta_max = np.inf
        self._gamma = 0.0
        self._s = np.zeros(self._param_sum)
        self._delta = self._delta_init / self._Delta
        self._eps = self._delta

    @property
    def n_continuous(self) -> int:
        """Number of continuous variables"""
        return self._Nco

    @property
    def n_integer(self) -> int:
        """Number of integer variables"""
        return self._Nin

    @property
    def n_categorical(self) -> int:
        """Number of categorical variables"""
        return self._Nca

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when the distribution is updated."""
        return self._g

    def reseed_rng(self, seed: int) -> None:
        """Reseeds the internal random number generator."""
        self._rng.seed(seed)

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D**2)), B.T)

        self._B, self._D = B, D
        return B, D

    def _sample_from_gaussian(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        xi = self._rng.randn(self._Nmi)  # ~ N(0, I)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(xi)  # ~ N(0, C)
        v = self._mean + self._sigma * self._A * y  # ~ N(m, σ^2 A C A)
        return v

    def _sample_from_categorical(self) -> np.ndarray:
        # Categorical variables are one-hot encoded.
        # Variables with fewer categories are zero-padded at the end.
        rand_q = self._rng.rand(self._Nca, 1)
        cum_q = self._q.cumsum(axis=1)
        c = (cum_q - self._q <= rand_q) & (rand_q < cum_q)
        return c

    def _repair_continuous_params(self, continuous_param: np.ndarray) -> np.ndarray:
        if self._x_space is None:
            return continuous_param

        # clip with lower and upper bound.
        param = np.where(
            continuous_param < self._x_space[:, 0],
            self._x_space[:, 0],
            continuous_param,
        )
        param = np.where(param > self._x_space[:, 1], self._x_space[:, 1], param)
        return param

    def _discretization(self, v_discrete: np.ndarray) -> np.ndarray:
        z_pos = np.array(
            [
                np.searchsorted(self._z_lim[i], v_discrete[i])
                for i in range(len(v_discrete))
            ]
        )
        z = self._z_space[np.arange(len(self._z_space)), z_pos]
        return z

    def _calc_continuous_penalty(
        self, v_raw: np.ndarray, sorted_fvals: np.ndarray
    ) -> np.ndarray:
        # penalty values for box constraint handling:
        # https://ieeexplore.ieee.org/document/4634579
        iq_range = (
            sorted_fvals[3 * self._popsize // 4] - sorted_fvals[self._popsize // 4]
        )

        # insert iq_range in history
        if np.isfinite(iq_range) and iq_range > 0:
            self._iqhist_values.insert(0, iq_range)
        elif iq_range == np.inf and len(self._iqhist_values) > 1:
            self._iqhist_values.insert(0, max(self._iqhist_values))
        else:
            pass  # ignore 0 or nan values

        if len(self._iqhist_values) > self._iqhist_term:
            self._iqhist_values.pop()

        bound_low = np.concatenate((self._x_space[:, 0], np.full(self._Nin, -np.inf)))
        bound_up = np.concatenate((self._x_space[:, 1], np.full(self._Nin, np.inf)))
        diag_CA = np.diag(self._C) * self._A

        delta_fit = np.median(self._iqhist_values)
        gamma = np.ones(self._Nmi) * 2 * delta_fit / (self._sigma**2 * np.sum(diag_CA))

        gamma_inc_low = (self._mean < bound_low) * (
            np.abs(self._mean - bound_low)
            > 3
            * self._sigma
            * np.sqrt(diag_CA)
            * max(1, np.sqrt(self._Nmi) / self._mu_eff)
        )
        gamma_inc_up = (bound_up < self._mean) * (
            np.abs(bound_up - self._mean)
            > 3
            * self._sigma
            * np.sqrt(diag_CA)
            * max(1, np.sqrt(self._Nmi) / self._mu_eff)
        )
        gamma_inc = np.logical_or(gamma_inc_low, gamma_inc_up)
        gamma[gamma_inc] *= 1.1 ** (max(1, self._mu_eff / (10 * self._Nmi)))

        xis = np.exp(0.9 * (np.log(diag_CA) - np.sum(np.log(diag_CA)) / self._Nmi))
        v_feas = np.where(
            v_raw < bound_low, bound_low, np.where(v_raw > bound_up, bound_up, v_raw)
        )
        penalties = np.sum(gamma * ((v_feas - v_raw) ** 2) / xis, axis=1)
        return penalties

    def _integer_centering(self, v_raw: np.ndarray) -> np.ndarray:
        # integer centering and
        # calculation of whether a successful integer mutation occurred
        v_old = np.copy(v_raw)
        int_m = self._discretization(self._mean[self._discrete_idx])
        mpos = np.zeros(self._Nin)
        mneg = np.zeros(self._Nin)
        self._int_succ = np.zeros(self._Nin, dtype=bool)
        for i in range(self._mu):
            vin_i = v_raw[i, self._discrete_idx]
            int_vin_i = self._discretization(vin_i)
            mutated = int_vin_i != int_m
            self._int_succ = np.logical_or(self._int_succ, mutated)
            mpos += (~mutated) * ((int_vin_i - vin_i) > 0) * (int_vin_i - vin_i)
            mneg += (~mutated) * ((int_vin_i - vin_i) < 0) * (int_vin_i - vin_i)
            v_raw[i, self._discrete_idx[mutated]] = int_vin_i[mutated]

        bias = np.sum((v_raw - v_old)[: self._mu, self._discrete_idx], axis=0)
        alphas = np.zeros(self._Nin)
        for moves in [mpos, mneg]:
            idx = bias * moves < 0
            alphas[idx] = np.minimum(1, -bias[idx] / moves[idx])

        for i in range(self._mu):
            int_voldin_i = self._discretization(v_old[i, self._discrete_idx])
            int_vin_i = self._discretization(v_raw[i, self._discrete_idx])
            Delta = int_voldin_i - v_old[i, self._discrete_idx]
            non_mutated = int_vin_i == int_m
            bias_Delta_cond = bias * Delta < 0
            indic = np.logical_and(bias_Delta_cond, non_mutated)
            v_raw[i, self._discrete_idx] += indic * alphas * Delta

        return v_raw

    def _update_gaussian(self, sv: np.ndarray) -> None:
        x_k = (sv - self._mean) / self._A + self._mean  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        B, D = self._eigen_decomposition()

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
        h_sigma_cond_right = (1.4 + 2 / (self._Nmi + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        w_io = self._weights * np.where(
            self._weights >= 0,
            1,
            self._Nmi / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)
        assert delta_h_sigma <= 1

        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
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

        # post-processing to prevent the minimum eigenvalue from becoming too small
        self._B, self._D = None, None
        B_updated, D_updated = self._eigen_decomposition()
        sigma_min = np.sqrt(self._min_eigenvalue / np.min(D_updated))
        self._sigma = max(self._sigma, sigma_min)

    def _margin_correction(self) -> None:
        updated_m_integer = self._mean[self._discrete_idx]

        # nearest discretization thresholds
        m_pos = np.array(
            [
                np.searchsorted(self._z_lim[i], updated_m_integer[i])
                for i in range(len(updated_m_integer))
            ]
        )
        z_lim_low_index = np.clip(m_pos - 1, 0, self._z_lim.shape[1] - 1)
        z_lim_up_index = np.clip(m_pos, 0, self._z_lim.shape[1] - 1)
        m_z_lim_low = self._z_lim[np.arange(len(self._z_lim)), z_lim_low_index]
        m_z_lim_up = self._z_lim[np.arange(len(self._z_lim)), z_lim_up_index]

        # low_cdf := Pr(X <= m_z_lim_low)
        # up_cdf := Pr(m_z_lim_up < X)
        z_scale = (
            self._sigma
            * self._A[self._discrete_idx]
            * np.sqrt(np.diag(self._C)[self._discrete_idx])
        )
        low_cdf = norm_cdf(m_z_lim_low, loc=updated_m_integer, scale=z_scale)
        up_cdf = 1.0 - norm_cdf(m_z_lim_up, loc=updated_m_integer, scale=z_scale)
        mid_cdf = 1.0 - (low_cdf + up_cdf)

        # edge case
        edge_mask = np.maximum(low_cdf, up_cdf) > 0.5
        # otherwise
        side_mask = np.maximum(low_cdf, up_cdf) <= 0.5
        # indices of successful integer mutations
        suc_idx = np.where(self._int_succ)
        nsuc_idx = np.where(~self._int_succ)

        if np.any(edge_mask):
            # modify sign
            modify_sign = np.sign(self._mean[self._discrete_idx] - m_z_lim_up)

            # clip mutation rates
            p_mut = np.minimum(low_cdf, up_cdf)
            p_mut = np.maximum(p_mut, self._alpha)
            p_mut[nsuc_idx] = np.minimum(p_mut[nsuc_idx], self._pmut[nsuc_idx])
            indices_to_update = self._discrete_idx[edge_mask]

            # avoid numerical errors
            p_mut = np.clip(p_mut, _EPS, 0.5 - _EPS)

            # modify A
            m_int = self._discretization(updated_m_integer)
            A_lower = np.abs(m_int - m_z_lim_up) / (
                self._sigma
                * np.sqrt(
                    chi2_ppf(q=1.0 - 2.0 * self._alpha)
                    * np.diag(self._C)[self._discrete_idx]
                )
            )
            self._A[indices_to_update] = np.maximum(
                self._A[indices_to_update], A_lower[edge_mask]
            )

            # distance from m_z_lim_up
            dist = (
                self._sigma
                * self._A[self._discrete_idx]
                * np.sqrt(
                    chi2_ppf(q=1.0 - 2.0 * p_mut) * np.diag(self._C)[self._discrete_idx]
                )
            )

            # modify mean vector
            self._mean[self._discrete_idx] = self._mean[
                self._discrete_idx
            ] + edge_mask * (
                m_z_lim_up + modify_sign * dist - self._mean[self._discrete_idx]
            )

            # save mutation rates for the next generation
            self._pmut[edge_mask] = p_mut[edge_mask]

        if np.any(side_mask):
            low_cdf = np.maximum(low_cdf, self._alpha / 2)
            up_cdf = np.maximum(up_cdf, self._alpha / 2)
            mid_cdf[nsuc_idx] = np.maximum(mid_cdf[nsuc_idx], 1 - self._pmut[nsuc_idx])

            Delta_cdf = 1 - low_cdf - up_cdf - mid_cdf

            Delta_cdf[suc_idx] /= (
                low_cdf[suc_idx]
                + up_cdf[suc_idx]
                + mid_cdf[suc_idx]
                - 3 * self._alpha / 2
            )
            Delta_cdf[nsuc_idx] /= (
                low_cdf[nsuc_idx]
                + up_cdf[nsuc_idx]
                + mid_cdf[nsuc_idx]
                - self._alpha
                - (1 - self._pmut[nsuc_idx])
            )

            low_cdf += Delta_cdf * (low_cdf - self._alpha / 2)
            up_cdf += Delta_cdf * (up_cdf - self._alpha / 2)

            # avoid numerical errors
            low_cdf = np.clip(low_cdf, _EPS, 0.5 - _EPS)
            up_cdf = np.clip(up_cdf, _EPS, 0.5 - _EPS)

            # modify mean vector and A (with sigma and C fixed)
            chi_low_sq = np.sqrt(chi2_ppf(q=1.0 - 2 * low_cdf))
            chi_up_sq = np.sqrt(chi2_ppf(q=1.0 - 2 * up_cdf))
            C_diag_sq = np.sqrt(np.diag(self._C))[self._discrete_idx]

            self._A[self._discrete_idx] = self._A[self._discrete_idx] + side_mask * (
                (m_z_lim_up - m_z_lim_low)
                / ((chi_low_sq + chi_up_sq) * self._sigma * C_diag_sq)
                - self._A[self._discrete_idx]
            )
            self._mean[self._discrete_idx] = self._mean[
                self._discrete_idx
            ] + side_mask * (
                (m_z_lim_low * chi_up_sq + m_z_lim_up * chi_low_sq)
                / (chi_low_sq + chi_up_sq)
                - self._mean[self._discrete_idx]
            )

            # save mutation rates for the next generation
            self._pmut[side_mask] = low_cdf[side_mask] + up_cdf[side_mask]

    def _update_categorical(self, sc: np.ndarray) -> None:
        # natural gradient
        ngrad = (
            self._weights[: self._mu, np.newaxis, np.newaxis]
            * (sc[: self._mu, :, :] - self._q)
        ).sum(axis=0)

        # approximation of the square root of the fisher information matrix:
        # Appendix B in https://proceedings.mlr.press/v97/akimoto19a.html
        sl = []
        for i, K in enumerate(self._K):
            q_i = self._q[i, : K - 1]
            q_i_K = self._q[i, K - 1]
            s_i = 1.0 / np.sqrt(q_i) * ngrad[i, : K - 1]
            s_i += np.sqrt(q_i) * ngrad[i, : K - 1].sum() / (q_i_K + np.sqrt(q_i_K))
            sl += list(s_i)
        ngrad_sqF = np.array(sl)

        pnorm = np.sqrt(np.dot(ngrad_sqF, ngrad_sqF))
        self._eps = self._delta / (pnorm + _EPS)
        self._q += self._eps * ngrad

        # update of ASNG
        self._delta = self._delta_init / self._Delta
        beta = self._delta / (self._param_sum**0.5)
        self._s = (1 - beta) * self._s + np.sqrt(beta * (2 - beta)) * ngrad_sqF / pnorm
        self._gamma = (1 - beta) ** 2 * self._gamma + beta * (2 - beta)
        self._Delta *= np.exp(
            beta * (self._gamma - np.dot(self._s, self._s) / self._alpha_snr)
        )
        self._Delta = min(self._Delta, self._Delta_max)

        # margin correction for categorical distribution
        for i in range(self._Nca):
            Ki = self._K[i]
            self._q[i, :Ki] = np.maximum(self._q[i, :Ki], self._qmin[i])
            q_sum = self._q[i, :Ki].sum()
            tmp = q_sum - self._qmin[i] * Ki
            self._q[i, :Ki] -= (q_sum - 1) * (self._q[i, :Ki] - self._qmin[i]) / tmp
            self._q[i, :Ki] /= self._q[i, :Ki].sum()

    def ask(self) -> CatCMAwM.Solution:
        """Sample a solution from the current search distribution.

        Returns:
            Solution: A sampled Solution object containing continuous (x),
            integer (z), and/or categorical (c) variables.
        """
        x = None
        z = None
        c = None
        v_raw = None
        if self._use_gaussian:
            v_raw = self._sample_from_gaussian()
            if self._use_continuous:
                x_raw = v_raw[self._continuous_idx]
                x = self._repair_continuous_params(x_raw)
            if self._use_integer:
                z = self._discretization(v_raw[self._discrete_idx])
        if self._use_categorical:
            c = self._sample_from_categorical()
        return CatCMAwM.Solution(x, z, c, v_raw)

    def tell(self, solutions: List[Tuple[CatCMAwM.Solution, float]]) -> None:
        """Tell evaluation values"""

        if len(solutions) != self._popsize:
            raise ValueError(
                f"Must tell population_size-length solutions: "
                f"expected {self._popsize}, but got {len(solutions)}."
            )

        solutions.sort(key=lambda s: s[1])
        fvals = np.stack([sol[1] for sol in solutions])

        # calculate penalty values for infeasible continuous solutions
        penalties = np.zeros(self._popsize)
        if self._use_continuous:
            v_raw = np.stack([cast(np.ndarray, sol[0]._v_raw) for sol in solutions])
            penalties = self._calc_continuous_penalty(v_raw, fvals)

        for i in range(self._popsize):
            solutions[i] = (solutions[i][0], solutions[i][1] + penalties[i])

        solutions.sort(key=lambda s: s[1])

        sv = None
        sc = None
        if self._use_gaussian:
            sv = np.stack([cast(np.ndarray, sol[0]._v_raw) for sol in solutions])
            assert np.all(
                np.abs(sv) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors."
        if self._use_categorical:
            sc = np.stack([cast(np.ndarray, sol[0].c) for sol in solutions])

        self._g += 1

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        if self._use_gaussian:
            funhist_idx = 2 * (self.generation % self._funhist_term)
            self._funhist_values[funhist_idx] = fvals[0]
            self._funhist_values[funhist_idx + 1] = fvals[-1]

        # integer centering
        if self._use_integer:
            assert sv is not None, "sv (sample from gaussian) must not be None."
            sv = self._integer_centering(sv)

        # --- update distribution parameters ---
        if self._use_gaussian:
            assert sv is not None, "sv (sample from gaussian) must not be None."
            self._update_gaussian(sv)

        if self._use_integer:
            self._margin_correction()

        if self._use_categorical:
            assert sc is not None, "sc (sample from categorical) must not be None."
            self._update_categorical(sc)

    def should_stop(self) -> bool:
        """Termination conditions specifically tailored for mixed-variable
        cases are not yet implemented. Currently, only standard CMA-ES conditions for
        Gaussian distributions are used."""
        if not self._use_gaussian:
            warnings.warn(
                "Termination conditions are only applicable for Gaussian distribution."
            )
            return False

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
        i = self.generation % self._Nmi
        if np.all(self._mean == self._mean + (0.1 * self._sigma * D[i] * B[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._tolconditioncov:
            return True

        return False
