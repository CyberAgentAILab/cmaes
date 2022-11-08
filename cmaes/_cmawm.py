import functools
import math
import numpy as np

from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

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


class CMAwM:
    """CMA-ES with Margin class with ask-and-tell interface.
    The code is adapted from https://github.com/EvoConJP/CMA-ES_with_Margin.

    Example:

        .. code::

            import numpy as np
            from cmaes import CMAwM

            def ellipsoid_onemax(x, n_zdim):
                n = len(x)
                n_rdim = n - n_zdim
                ellipsoid = sum([(1000 ** (i / (n_rdim - 1)) * x[i]) ** 2 for i in range(n_rdim)])
                onemax = n_zdim - (0. < x[(n - n_zdim):]).sum()
                return ellipsoid + 10 * onemax

            binary_dim, continuous_dim = 10, 10
            dim = binary_dim + continuous_dim
            bounds = np.concatenate(
                [
                    np.tile([0, 1], (binary_dim, 1)),
                    np.tile([-np.inf, np.inf], (continuous_dim, 1)),
                ]
            )
            steps = np.concatenate([np.ones(binary_dim), np.zeros(continuous_dim)])
            optimizer = CMAwM(mean=np.zeros(dim), sigma=2.0, bounds=bounds, steps=steps)

            evals = 0
            while True:
                solutions = []
                for _ in range(optimizer.population_size):
                    x_for_eval, x_for_tell = optimizer.ask()
                    value = ellipsoid_onemax(x_for_eval, binary_dim)
                    evals += 1
                    solutions.append((x_for_tell, value))
                optimizer.tell(solutions)

                if optimizer.should_stop():
                    break

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter.

        steps:
            Each value represents a step of discretization for each dimension.
            Zero (or negative value) means a continuous space.

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

    # Paper: https://arxiv.org/abs/2205.13482

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: np.ndarray,
        steps: np.ndarray,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
        margin: Optional[float] = None,
    ):
        assert len(bounds) == len(steps), "bounds and steps must be the same length"
        assert not np.isnan(steps).any(), "steps should not include NaN"

        # split discrete space and continuous space
        self._discrete_idx = np.where(steps > 0)[0]
        self._continuous_idx = np.where(steps <= 0)[0]
        assert (
            len(self._discrete_idx) > 0
        ), """steps should include at least one positive values corresponding to discrete
        dimensions."""
        discrete_list = [
            np.arange(bounds[i][0], bounds[i][1] + steps[i] / 2, steps[i])
            for i in self._discrete_idx
        ]
        max_discrete = max(len(discrete) for discrete in discrete_list)
        discrete_space = np.full((len(self._discrete_idx), max_discrete), np.nan)
        for i, discrete in enumerate(discrete_list):
            discrete_space[i, : len(discrete)] = discrete
        continuous_space = bounds[self._continuous_idx]

        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        n_zdim = len(discrete_space)
        n_rdim = n_dim - n_zdim
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))  # (eq. 48)
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # (eq.49)
        weights_prime = np.array(
            [
                math.log((population_size + 1) / 2) - math.log(i + 1)
                for i in range(population_size)
            ]
        )
        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(
            weights_prime[mu:] ** 2
        )

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

        min_alpha = min(
            1 + c1 / cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq.51
            (1 - c1 - cmu) / (n_dim * cmu),  # eq.52
        )

        # (eq.53)
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control (eq.55)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
            c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        margin = margin if margin is not None else 1 / (n_dim * population_size)
        assert margin > 0, "margin must be non-zero positive value."

        self._n_dim = n_dim
        self._n_zdim = n_zdim
        self._n_rdim = n_rdim
        self._popsize = population_size
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
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

        # continuous_space contains low and high of each parameter.
        assert _is_valid_bounds(
            continuous_space, mean[self._continuous_idx]
        ), "invalid bounds"
        self._continuous_space = continuous_space
        self._n_max_resampling = n_max_resampling

        # discrete_space
        self.margin = (
            margin if margin is not None else 1 / (self._n_dim * self._popsize)
        )
        self.z_space = discrete_space
        self.z_lim = (self.z_space[:, 1:] + self.z_space[:, :-1]) / 2
        for i in range(self._n_zdim):
            self.z_space[i][np.isnan(self.z_space[i])] = np.nanmax(self.z_space[i])
            self.z_lim[i][np.isnan(self.z_lim[i])] = np.nanmax(self.z_lim[i])
        self.z_lim_low = np.concatenate(
            [self.z_lim.min(axis=1).reshape([self._n_zdim, 1]), self.z_lim], 1
        )
        self.z_lim_up = np.concatenate(
            [self.z_lim, self.z_lim.max(axis=1).reshape([self._n_zdim, 1])], 1
        )
        m_z = self._mean[self._discrete_idx].reshape(([self._n_zdim, 1]))
        # m_z_lim_low ->|  mean vector    |<- m_z_lim_up
        self.m_z_lim_low = (
            self.z_lim_low
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)
        self.m_z_lim_up = (
            self.z_lim_up
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)

        self._A = np.full(self._n_dim, 1.0)

        self._g = 0
        self._rng = np.random.RandomState(seed)

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    def __getstate__(self) -> Dict[str, Any]:
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

    def __setstate__(self, state: Dict[str, Any]) -> None:
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

    def ask(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a parameter and return (i) encoded x and (ii) raw x.
        The encoded x is used for the evaluation.
        The raw x is used for updating the distribution."""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_continuous_feasible(x[self._continuous_idx]):
                x_encoded = x.copy()
                x_encoded[self._discrete_idx] = self._encoding_discrete_params(
                    x[self._discrete_idx]
                )
                return x_encoded, x
        x = self._sample_solution()
        x_encoded = x.copy()
        x_encoded[self._continuous_idx] = self._repair_continuous_params(
            x[self._continuous_idx]
        )
        x_encoded[self._discrete_idx] = self._encoding_discrete_params(
            x[self._discrete_idx]
        )
        return x_encoded, x

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def _is_continuous_feasible(self, continuous_param: np.ndarray) -> bool:
        if self._continuous_space is None:
            return True
        return cast(
            bool,
            np.all(continuous_param >= self._continuous_space[:, 0])
            and np.all(continuous_param <= self._continuous_space[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_continuous_params(self, continuous_param: np.ndarray) -> np.ndarray:
        if self._continuous_space is None:
            return continuous_param

        # clip with lower and upper bound.
        param = np.where(
            continuous_param < self._continuous_space[:, 0],
            self._continuous_space[:, 0],
            continuous_param,
        )
        param = np.where(
            param > self._continuous_space[:, 1], self._continuous_space[:, 1], param
        )
        return param

    def _encoding_discrete_params(self, discrete_param: np.ndarray) -> np.ndarray:
        """Encode the values into discrete domain."""
        x = (discrete_param - self._mean[self._discrete_idx]) * self._A[
            self._discrete_idx
        ] + self._mean[self._discrete_idx]
        x = x.reshape([self._n_zdim, 1])
        x_enc = (
            self.z_space
            * np.where(np.sort(np.concatenate((self.z_lim, x), axis=1)) == x, 1, 0)
        ).sum(axis=1)
        return x_enc.reshape(self._n_zdim)

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

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
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
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
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # (eq.45)
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        # (eq.46)
        w_io = self._weights * np.where(
            self._weights >= 0,
            1,
            self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # (eq.47)
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

        # margin correction if margin > 0
        if self.margin > 0:
            updated_m_integer = self._mean[self._discrete_idx, np.newaxis]
            self.z_lim_low = np.concatenate(
                [self.z_lim.min(axis=1).reshape([self._n_zdim, 1]), self.z_lim], 1
            )
            self.z_lim_up = np.concatenate(
                [self.z_lim, self.z_lim.max(axis=1).reshape([self._n_zdim, 1])], 1
            )
            self.m_z_lim_low = (
                self.z_lim_low
                * np.where(
                    np.sort(np.concatenate([self.z_lim, updated_m_integer], 1))
                    == updated_m_integer,
                    1,
                    0,
                )
            ).sum(axis=1)
            self.m_z_lim_up = (
                self.z_lim_up
                * np.where(
                    np.sort(np.concatenate([self.z_lim, updated_m_integer], 1))
                    == updated_m_integer,
                    1,
                    0,
                )
            ).sum(axis=1)

            # calculate probability low_cdf := Pr(X <= m_z_lim_low) and up_cdf := Pr(m_z_lim_up < X)
            # sig_z_sq_Cdiag = self.model.sigma * self.model.A * np.sqrt(np.diag(self.model.C))
            z_scale = (
                self._sigma
                * self._A[self._discrete_idx]
                * np.sqrt(np.diag(self._C)[self._discrete_idx])
            )
            updated_m_integer = updated_m_integer.flatten()
            low_cdf = norm_cdf(self.m_z_lim_low, loc=updated_m_integer, scale=z_scale)
            up_cdf = 1.0 - norm_cdf(
                self.m_z_lim_up, loc=updated_m_integer, scale=z_scale
            )
            mid_cdf = 1.0 - (low_cdf + up_cdf)
            # edge case
            edge_mask = np.maximum(low_cdf, up_cdf) > 0.5
            # otherwise
            side_mask = np.maximum(low_cdf, up_cdf) <= 0.5

            if np.any(edge_mask):
                # modify mask (modify or not)
                modify_mask = np.minimum(low_cdf, up_cdf) < self.margin
                # modify sign
                modify_sign = np.sign(self._mean[self._discrete_idx] - self.m_z_lim_up)
                # distance from m_z_lim_up
                dist = (
                    self._sigma
                    * self._A[self._discrete_idx]
                    * np.sqrt(
                        chi2_ppf(q=1.0 - 2.0 * self.margin)
                        * np.diag(self._C)[self._discrete_idx]
                    )
                )
                # modify mean vector
                self._mean[self._discrete_idx] = self._mean[
                    self._discrete_idx
                ] + modify_mask * edge_mask * (
                    self.m_z_lim_up
                    + modify_sign * dist
                    - self._mean[self._discrete_idx]
                )

            # correct probability
            low_cdf = np.maximum(low_cdf, self.margin / 2.0)
            up_cdf = np.maximum(up_cdf, self.margin / 2.0)
            modified_low_cdf = low_cdf + (1.0 - low_cdf - up_cdf - mid_cdf) * (
                low_cdf - self.margin / 2
            ) / (low_cdf + mid_cdf + up_cdf - 3.0 * self.margin / 2)
            modified_up_cdf = up_cdf + (1.0 - low_cdf - up_cdf - mid_cdf) * (
                up_cdf - self.margin / 2
            ) / (low_cdf + mid_cdf + up_cdf - 3.0 * self.margin / 2)
            modified_low_cdf = np.clip(modified_low_cdf, 1e-10, 0.5 - 1e-10)
            modified_up_cdf = np.clip(modified_up_cdf, 1e-10, 0.5 - 1e-10)

            # modify mean vector and A (with sigma and C fixed)
            chi_low_sq = np.sqrt(chi2_ppf(q=1.0 - 2 * modified_low_cdf))
            chi_up_sq = np.sqrt(chi2_ppf(q=1.0 - 2 * modified_up_cdf))
            C_diag_sq = np.sqrt(np.diag(self._C))[self._discrete_idx]

            # simultaneous equations
            self._A[self._discrete_idx] = self._A[self._discrete_idx] + side_mask * (
                (self.m_z_lim_up - self.m_z_lim_low)
                / ((chi_low_sq + chi_up_sq) * self._sigma * C_diag_sq)
                - self._A[self._discrete_idx]
            )
            self._mean[self._discrete_idx] = self._mean[
                self._discrete_idx
            ] + side_mask * (
                (self.m_z_lim_low * chi_up_sq + self.m_z_lim_up * chi_low_sq)
                / (chi_low_sq + chi_up_sq)
                - self._mean[self._discrete_idx]
            )

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
