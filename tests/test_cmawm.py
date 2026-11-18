import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase
from cmaes import CMA, CMAwM


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

    def test_discrete_encoding_with_large_normalized_samples(self):
        step = 2.0**-68
        optimizer = CMAwM(
            mean=np.zeros(2),
            sigma=1.0,
            bounds=np.array([[-step, step], [-np.inf, np.inf]]),
            steps=np.array([step, 0.0]),
            seed=1,
        )

        with np.errstate(invalid="raise"):
            for _ in range(2):
                encoded, raw = optimizer.ask()
                expected = step if raw[0] > step / 2 else -step if raw[0] <= -step / 2 else 0.0
                self.assertEqual(encoded[0], expected)
