from __future__ import annotations

import math
import gpytorch.distributions
import numpy as np

from typing import Any
from typing import cast
from typing import Optional

import scipy
import gpytorch
import torch

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class SafeCMA:
    """Safe CMA-ES stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

            import numpy as np
            from cmaes import SafeCMA

            # number of dimensions
            dim = 5

            # objective function
            def quadratic(x):
                coef = 1000 ** (np.arange(dim) / float(dim - 1))
                return np.sum((x * coef) ** 2)

            # safety function
            def safe_function(x):
                return x[0]

            # safe seeds
            safe_seeds_num = 10
            safe_seeds = (np.random.rand(safe_seeds_num, dim) * 2 - 1) * 5
            safe_seeds[:, 0] = - np.abs(safe_seeds[:, 0])

            # evaluation of safe seeds (with a single safety function)
            seeds_evals = np.array([quadratic(x) for x in safe_seeds])
            seeds_safe_evals = np.stack([[safe_function(x)] for x in safe_seeds])
            safety_threshold = np.array([0])

            # optimizer (safe CMA-ES)
            optimizer = SafeCMA(
                sigma=1.,
                safety_threshold=safety_threshold,
                safe_seeds=safe_seeds,
                seeds_evals=seeds_evals,
                seeds_safe_evals=seeds_safe_evals,
            )

            unsafe_eval_counts = 0
            best_eval = np.inf

            for generation in range(400):
                solutions = []
                for _ in range(optimizer.population_size):
                    # Ask a parameter
                    x = optimizer.ask()
                    value = quadratic(x)
                    safe_value = np.array([safe_function(x)])

                    # save best eval
                    best_eval = np.min((best_eval, value))
                    unsafe_eval_counts += (safe_value > safety_threshold)

                    solutions.append((x, value, safe_value))

                # Tell evaluation values.
                optimizer.tell(solutions)

                print(f"#{generation} ({best_eval} {unsafe_eval_counts})")

                if optimizer.should_stop():
                    break

    Args:

        safe_seeds:
            Solutions whose safe function values are above the safety thresholds.
            Safe CMA-ES uses the safe seed with the best evaluation value as
            the initial mean vector of multi-variate Gaussian distributions.

        seeds_evals:
            Evaluation values of safe seeds on the objective function.

        seeds_safe_evals:
            Evaluation values of safe seeds on the safe functions.

        safety_threshold:
            Safety thresholds for each safe functions.

        sigma:
            Initial standard deviation of covariance matrix.
            Safe CMA-ES modifies sigma when more than two safe seeds are given.

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

    # Paper: https://arxiv.org/abs/2405.10534

    def __init__(
        self,
        safe_seeds: np.ndarray,
        seeds_evals: np.ndarray,
        seeds_safe_evals: np.ndarray,
        safety_threshold: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
    ):
        # safety threshold
        self.safety_threshold = safety_threshold
        self.safety_func_num = len(safety_threshold)

        # safe seeds
        self.safe_seeds = safe_seeds
        self.seeds_evals = seeds_evals
        self.seeds_safe_evals = seeds_safe_evals

        n_dim = len(safe_seeds[0])
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = population_size // 2

        # hyperparameters for safe CMAES
        self.kernel = gpytorch.kernels.RBFKernel()
        self.kernel.lengthscale = 8.0 * n_dim

        self.lip_penalty_coef = 1.0
        self.lip_penalty_inc_rate = 10  # alpha
        self.lip_penalty_dec_rate = self.lip_penalty_inc_rate ** (1.0 / n_dim)

        self.lip_ite = 5  # T_data
        self.sample_num_lip = population_size * self.lip_ite
        self.sample_log_num = population_size * self.lip_ite
        self.init_L_base = 10  # zeta_init
        self.init_L = 100
        self.gamma = 0.9

        # log for safe CMAES
        self.sampled_points = safe_seeds.copy()
        self.sampled_safe_evals = seeds_safe_evals.copy()

        # safe CMA-ES do not use negative weights
        weights_prime = np.array(
            np.log((population_size + 1) / 2) - np.log(np.arange(population_size) + 1)
        )
        weights_prime[weights_prime < 0] = 0

        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
        weights = weights_prime / weights_prime.sum()

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

        if cov is None:
            self._C = np.eye(n_dim)
        else:
            assert cov.shape == (n_dim, n_dim), "Invalid shape of covariance matrix"
            self._C = cov

        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        self._rng = np.random.RandomState(seed)

        # initial distribution parameter
        self._sigma = sigma
        mean, sigma = self._init_distribution(sigma)

        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        self._mean = mean.copy()
        self._sigma = sigma

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    def _compute_lipschitz_constant(self) -> np.ndarray:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(0)
        )
        likelihood.noise = 0

        B, D = self._eigen_decomposition()
        invSqrtC = cast(np.ndarray, B.dot(np.diag(1 / D)).dot(B.T))

        num_data = int(np.min((len(self.sampled_safe_evals), self.sample_num_lip)))
        prev_x = self.sampled_points[-num_data:]
        z_points = (prev_x - self._mean).dot(invSqrtC) / self._sigma

        target_safe_evals = self.sampled_safe_evals[-num_data:]
        evals_mean = np.mean(target_safe_evals, axis=0)
        evals_std = np.std(target_safe_evals, axis=0)
        modified_evals = (target_safe_evals - evals_mean) / evals_std

        # function that returns the negative norm of gradient
        def df(x: np.ndarray, model: ExactGPModel) -> torch.Tensor:
            out_scalar = x.ndim == 1
            x = np.atleast_2d(x)

            grad_norm = torch.zeros(len(x))

            X = torch.autograd.Variable(
                torch.Tensor(np.atleast_2d(x)), requires_grad=True
            )
            mean = likelihood(model(X)).mean
            dxdmean = torch.autograd.grad(mean.sum(), X)[0]

            grad_norm = torch.sqrt(torch.sum(dxdmean * dxdmean, dim=1))

            if out_scalar:
                grad_norm = grad_norm.mean().to(torch.float64)

            return -grad_norm

        def elementwise_df(i: int) -> float:
            samples = self._rng.randn(self.sample_num_lip, self._n_dim)
            samples = np.concatenate([samples, z_points], axis=0)
            model = ExactGPModel(
                z_points, modified_evals[:, i], likelihood, self.kernel
            )

            try:
                pred_samples = df(samples, model) * evals_std[i]
            except Exception:
                # if fail to optimize
                return self.lipschitz_constant[i]

            if np.isnan(pred_samples).any():
                return self.lipschitz_constant[i]

            x0 = samples[np.argmin(pred_samples)]

            try:
                bounds = np.tile([-3, 3], (self._n_dim, 1))

                res = scipy.optimize.minimize(
                    df,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    args=(model),
                    options={"maxiter": 200},
                )
                result_value = res.fun * evals_std[i]

                if not np.isnan(result_value):
                    return -float(result_value)
                else:
                    return -np.min(pred_samples)
            except Exception:
                # if fail to optimize
                return -np.min(pred_samples)

        return np.array([elementwise_df(i) for i in range(self.safety_func_num)])

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

    def _init_distribution(self, sigma: float) -> tuple[np.ndarray, float]:
        # set initial mean vector
        best_seed_id = np.argmin(self.seeds_evals)
        mean = self.safe_seeds[best_seed_id]
        self._mean = mean.copy()  # (eq. 26)

        # set initial step-size
        if len(self.sampled_points) > 1:
            lip = self._compute_lipschitz_constant()

            if len(self.sampled_safe_evals) < self.sample_num_lip:
                exponent = 1 / len(self.sampled_safe_evals)
                lip = lip * (self.init_L_base**exponent)

            lip = np.clip(lip, self.init_L, None)
        else:
            lip = np.ones(self.safety_func_num) * self.init_L

        self.lipschitz_constant = lip

        slack = self.safety_threshold - self.seeds_safe_evals[best_seed_id]
        delta = np.min((slack) / self.lipschitz_constant)
        gauss_tr = np.sqrt(scipy.stats.chi2.ppf(self.gamma, df=self._n_dim))
        sigma = sigma * np.min((delta / gauss_tr, 1))  # (eq. 27)

        return mean, sigma

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

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
        invSqrtC = cast(np.ndarray, B.dot(np.diag(1 / D)).dot(B.T))

        if self.sampled_safe_evals is not None:
            log_num = np.min([self.sample_log_num, len(self.sampled_points)])
            prev_x = self.sampled_points[-log_num:]
            prev_safe_evals = self.sampled_safe_evals[-log_num:]
            sampled_z_points = (prev_x - self._mean).dot(invSqrtC) / self._sigma

            # radius: radius of trust region around evaluated points
            slack = self.safety_threshold[:, None, None] - prev_safe_evals[None, :, :]
            radius = np.min(
                slack / self.lipschitz_constant[:, None, None], axis=(0, 2)
            )  # (eq.13)

            radius[radius < 0] = -np.inf
            # dist: distance between current samples and evaluated points
            dist = np.sqrt(((z[None, :] - sampled_z_points) ** 2).sum(axis=1))

            invalid_dist = np.clip(np.min(dist[None, :] - radius), 0, np.inf)
            argmin_sample_id = np.argmin(dist[None, :] - radius)
            closest_z_sample = sampled_z_points[argmin_sample_id]

            ratio = invalid_dist / dist[argmin_sample_id]
            z = (1 - ratio) * z + ratio * closest_z_sample  # (eq.15)

        y = cast(np.ndarray, B.dot(np.diag(D)).dot(B.T)).dot(z)  # ~ N(0, C)
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

    def tell(self, solutions: list[tuple[np.ndarray, float, float]]) -> None:
        self._naive_cma_update(solutions)

        X = np.stack([s[0] for s in solutions])
        safe_evals = np.array([s[2] for s in solutions])

        self._add_evaluated_point(X, safe_evals)

        self.lipschitz_constant = self._compute_lipschitz_constant()  # (eq.19)
        if len(self.sampled_safe_evals) < self.sample_num_lip:
            exponent = 1 / len(self.sampled_safe_evals)  # (eq.22)
            self.lipschitz_constant *= self.init_L_base**exponent

        inv_num = float(np.sum(safe_evals > self.safety_threshold))
        if inv_num > 0:
            self.lip_penalty_coef *= self.lip_penalty_inc_rate ** (
                inv_num / self._popsize
            )
        else:
            self.lip_penalty_coef /= self.lip_penalty_dec_rate
            self.lip_penalty_coef = np.max((self.lip_penalty_coef, 1))
        self.lipschitz_constant *= self.lip_penalty_coef  # (eq.24)

    def _add_evaluated_point(self, X: np.ndarray, safe_evals: np.ndarray) -> None:
        self.sampled_points = np.concatenate([self.sampled_points, X], axis=0)
        self.sampled_safe_evals = np.vstack([self.sampled_safe_evals, safe_evals])

    def _naive_cma_update(
        self, solutions: list[tuple[np.ndarray, float, float]]
    ) -> None:
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
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)
        self._mean += self._cm * self._sigma * y_w  # (eq.7)

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
        )  # (eq.8)
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

        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(self._weights, y_k)]), axis=0
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)
        assert delta_h_sigma <= 1

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
        )  # (eq.9)

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


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel,
    ) -> None:
        super(ExactGPModel, self).__init__(
            torch.from_numpy(train_x), torch.from_numpy(train_y), likelihood
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

        self.eval()
        likelihood.eval()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
