from __future__ import annotations

import functools
import numpy as np

from typing import cast
from typing import Optional


from cmaes import CMA
from cmaes._cma import _is_valid_bounds

try:
    from scipy import stats

    chi2_ppf = functools.partial(stats.chi2.ppf, df=1)
    norm_cdf = stats.norm.cdf
except ImportError:
    from cmaes._stats import chi2_ppf  # type: ignore
    from cmaes._stats import norm_cdf


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
        # initialize `CMA`
        self._cma = CMA(
            mean, sigma, bounds, n_max_resampling, seed, population_size, cov
        )
        n_dim = self._cma.dim
        population_size = self._cma.population_size
        self._n_max_resampling = n_max_resampling

        # split discrete space and continuous space
        assert len(bounds) == len(steps), "bounds and steps must be the same length"
        assert not np.isnan(steps).any(), "steps should not include NaN"
        self._discrete_idx = np.where(steps > 0)[0]
        discrete_list = [
            np.arange(bounds[i][0], bounds[i][1] + steps[i] / 2, steps[i])
            for i in self._discrete_idx
        ]
        max_discrete = max([len(discrete) for discrete in discrete_list], default=0)
        discrete_space = np.full((len(self._discrete_idx), max_discrete), np.nan)
        for i, discrete in enumerate(discrete_list):
            discrete_space[i, : len(discrete)] = discrete

        # continuous_space contains low and high of each parameter.
        self._continuous_idx = np.where(steps <= 0)[0]
        self._continuous_space = bounds[self._continuous_idx]
        assert _is_valid_bounds(
            self._continuous_space, mean[self._continuous_idx]
        ), "invalid bounds"

        # discrete_space
        self._n_zdim = len(discrete_space)
        if self._n_zdim == 0:
            return
        self.margin = margin if margin is not None else 1 / (n_dim * population_size)
        assert self.margin > 0, "margin must be non-zero positive value."
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
        m_z = self._cma._mean[self._discrete_idx].reshape(([self._n_zdim, 1]))
        # m_z_lim_low ->|  mean vector    |<- m_z_lim_up
        self.m_z_lim_low = (
            self.z_lim_low
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)
        self.m_z_lim_up = (
            self.z_lim_up
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)

        self._A = np.full(n_dim, 1.0)

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._cma.dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._cma.population_size

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._cma.generation

    @property
    def mean(self) -> np.ndarray:
        """Mean Vector"""
        return self._cma.mean

    @property
    def _rng(self) -> np.random.RandomState:
        return self._cma._rng

    def reseed_rng(self, seed: int) -> None:
        self._cma.reseed_rng(seed)

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a parameter and return (i) encoded x and (ii) raw x.
        The encoded x is used for the evaluation.
        The raw x is used for updating the distribution."""
        for i in range(self._n_max_resampling):
            x = self._cma._sample_solution()
            if self._is_continuous_feasible(x[self._continuous_idx]):
                x_encoded = x.copy()
                if self._n_zdim > 0:
                    x_encoded[self._discrete_idx] = self._encode_discrete_params(
                        x[self._discrete_idx]
                    )
                return x_encoded, x
        x = self._cma._sample_solution()
        x[self._continuous_idx] = self._repair_continuous_params(
            x[self._continuous_idx]
        )
        x_encoded = x.copy()
        if self._n_zdim > 0:
            x_encoded[self._discrete_idx] = self._encode_discrete_params(
                x[self._discrete_idx]
            )
        return x_encoded, x

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

    def _encode_discrete_params(self, discrete_param: np.ndarray) -> np.ndarray:
        """Encode the values into discrete domain."""
        mean = self._cma._mean

        x = (discrete_param - mean[self._discrete_idx]) * self._A[
            self._discrete_idx
        ] + mean[self._discrete_idx]
        x = x.reshape([self._n_zdim, 1])
        x_enc = (
            self.z_space
            * np.where(np.sort(np.concatenate((self.z_lim, x), axis=1)) == x, 1, 0)
        ).sum(axis=1)
        return x_enc.reshape(self._n_zdim)

    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""
        self._cma.tell(solutions)
        mean = self._cma._mean
        sigma = self._cma._sigma
        C = self._cma._C

        if self._n_zdim == 0:
            return
        # margin correction
        updated_m_integer = mean[self._discrete_idx, np.newaxis]
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
            sigma
            * self._A[self._discrete_idx]
            * np.sqrt(np.diag(C)[self._discrete_idx])
        )
        updated_m_integer = updated_m_integer.flatten()
        low_cdf = norm_cdf(self.m_z_lim_low, loc=updated_m_integer, scale=z_scale)
        up_cdf = 1.0 - norm_cdf(self.m_z_lim_up, loc=updated_m_integer, scale=z_scale)
        mid_cdf = 1.0 - (low_cdf + up_cdf)
        # edge case
        edge_mask = np.maximum(low_cdf, up_cdf) > 0.5
        # otherwise
        side_mask = np.maximum(low_cdf, up_cdf) <= 0.5

        if np.any(edge_mask):
            # modify mask (modify or not)
            modify_mask = np.minimum(low_cdf, up_cdf) < self.margin
            # modify sign
            modify_sign = np.sign(mean[self._discrete_idx] - self.m_z_lim_up)
            # distance from m_z_lim_up
            dist = (
                sigma
                * self._A[self._discrete_idx]
                * np.sqrt(
                    chi2_ppf(q=1.0 - 2.0 * self.margin) * np.diag(C)[self._discrete_idx]
                )
            )
            # modify mean vector
            mean[self._discrete_idx] = mean[
                self._discrete_idx
            ] + modify_mask * edge_mask * (
                self.m_z_lim_up + modify_sign * dist - mean[self._discrete_idx]
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
        C_diag_sq = np.sqrt(np.diag(C))[self._discrete_idx]

        # simultaneous equations
        self._A[self._discrete_idx] = self._A[self._discrete_idx] + side_mask * (
            (self.m_z_lim_up - self.m_z_lim_low)
            / ((chi_low_sq + chi_up_sq) * sigma * C_diag_sq)
            - self._A[self._discrete_idx]
        )
        mean[self._discrete_idx] = mean[self._discrete_idx] + side_mask * (
            (self.m_z_lim_low * chi_up_sq + self.m_z_lim_up * chi_low_sq)
            / (chi_low_sq + chi_up_sq)
            - mean[self._discrete_idx]
        )

    def should_stop(self) -> bool:
        return self._cma.should_stop()
