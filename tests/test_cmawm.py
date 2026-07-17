import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_allclose
from unittest import TestCase
from cmaes import CMA, CMAwM


class _DenseCMAwM(CMAwM):
    """CMAwM using the dense discretization tables from the previous implementation."""

    def __init__(self, *args, **kwargs):
        bounds = kwargs["bounds"]
        steps = kwargs["steps"]
        discrete_idx = np.where(steps > 0)[0]
        discrete_list = [
            np.arange(bounds[i][0], bounds[i][1] + steps[i] / 2, steps[i])
            for i in discrete_idx
        ]
        max_discrete = max([len(discrete) for discrete in discrete_list], default=0)
        self._dense_z_space = np.full((len(discrete_idx), max_discrete), np.nan)
        for i, discrete in enumerate(discrete_list):
            self._dense_z_space[i, : len(discrete)] = discrete
        self._dense_z_lim = (
            self._dense_z_space[:, 1:] + self._dense_z_space[:, :-1]
        ) / 2
        for i in range(len(discrete_idx)):
            self._dense_z_space[i][np.isnan(self._dense_z_space[i])] = np.nanmax(
                self._dense_z_space[i]
            )
            self._dense_z_lim[i][np.isnan(self._dense_z_lim[i])] = np.nanmax(
                self._dense_z_lim[i]
            )
        super().__init__(*args, **kwargs)

    def _get_discrete_param_indices(self, values):
        return np.array(
            [np.searchsorted(self._dense_z_lim[i], values[i]) for i in range(len(values))]
        )

    def _get_discrete_param_values(self, indices):
        return self._dense_z_space[np.arange(len(indices)), indices]

    def _get_discrete_param_limit_values(self, indices):
        return self._dense_z_lim[np.arange(len(indices)), indices]


class TestCMAwM(TestCase):
    def test_no_discrete_spaces(self):
        mean = np.zeros(2)
        bounds = np.array([[-10, 10], [-10, 10]])
        steps = np.array([0, 0])
        sigma = 1.3
        seed = 1

        cma_optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            cmawm_optimizer = CMAwM(mean=mean, sigma=sigma, bounds=bounds, steps=steps, seed=seed)

        for i in range(100):
            solutions = []
            for _ in range(cma_optimizer.population_size):
                cma_x = cma_optimizer.ask()
                cmawm_x_encoded, cmawm_x_for_tell = cmawm_optimizer.ask()
                assert_almost_equal(cma_x, cmawm_x_encoded)
                assert_almost_equal(cma_x, cmawm_x_for_tell)

                objective = (cma_x[0] - 3) ** 2 + cma_x[1] ** 2
                solutions.append((cma_x, objective))
            cma_optimizer.tell(solutions)
            cmawm_optimizer.tell(solutions)

    def test_sampling_is_equivalent_to_dense_discretization(self):
        mean = np.array([0.0, 0.0, 0.0])
        bounds = np.array([[-30.0, 30.0], [-4.0, 4.0], [-5.0, 5.0]])
        steps = np.array([0.001, 0.5, 0.0])
        kwargs = {
            "mean": mean,
            "sigma": 1.3,
            "bounds": bounds,
            "steps": steps,
            "seed": 1,
        }
        optimizer = CMAwM(**kwargs)
        dense_optimizer = _DenseCMAwM(**kwargs)

        for _ in range(30):
            solutions = []
            dense_solutions = []
            for _ in range(optimizer.population_size):
                x_for_eval, x_for_tell = optimizer.ask()
                dense_x_for_eval, dense_x_for_tell = dense_optimizer.ask()

                assert_allclose(x_for_eval, dense_x_for_eval, rtol=0, atol=1e-12)
                assert_allclose(x_for_tell, dense_x_for_tell, rtol=0, atol=1e-12)

                # Pass the same objective value to both optimizers so this test isolates
                # differences in sampling and discretization from objective round-off.
                value = float(np.sum(x_for_eval**2))
                solutions.append((x_for_tell, value))
                dense_solutions.append((x_for_tell, value))

            optimizer.tell(solutions)
            dense_optimizer.tell(dense_solutions)
            assert_allclose(optimizer.mean, dense_optimizer.mean, rtol=0, atol=1e-12)
            assert_allclose(optimizer._A, dense_optimizer._A, rtol=0, atol=1e-12)

    def test_discrete_encoding_is_equivalent_to_dense_discretization(self):
        bounds = np.array([[-30.0, 30.0], [-4.0, 4.0], [-1.0, 1.0]])
        steps = np.array([0.001, 0.5, 0.3])
        kwargs = {
            "mean": np.zeros(3),
            "sigma": 1.3,
            "bounds": bounds,
            "steps": steps,
            "seed": 1,
        }
        optimizer = CMAwM(**kwargs)
        dense_optimizer = _DenseCMAwM(**kwargs)
        rng = np.random.RandomState(0)

        for _ in range(1000):
            values = rng.uniform(bounds[:, 0] - 1, bounds[:, 1] + 1)
            indices = optimizer._get_discrete_param_indices(values)
            dense_indices = dense_optimizer._get_discrete_param_indices(values)
            assert_allclose(
                optimizer._get_discrete_param_values(indices),
                dense_optimizer._get_discrete_param_values(dense_indices),
                rtol=0,
                atol=1e-12,
            )

        for dimension, size in enumerate(optimizer._discrete_space_size):
            for limit_index in (0, size // 2, size - 2):
                values = np.zeros(3)
                values[dimension] = dense_optimizer._dense_z_lim[dimension, limit_index]
                indices = optimizer._get_discrete_param_indices(values)
                dense_indices = dense_optimizer._get_discrete_param_indices(values)
                assert_allclose(
                    optimizer._get_discrete_param_values(indices),
                    dense_optimizer._get_discrete_param_values(dense_indices),
                    rtol=0,
                    atol=1e-12,
                )
