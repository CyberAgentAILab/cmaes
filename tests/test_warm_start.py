import numpy as np
import pytest
from unittest import TestCase
from cmaes import CMA, get_warm_start_mgd


class TestWarmStartCMA(TestCase):
    def test_dimension(self):
        optimizer = CMA(mean=np.zeros(10), sigma=1.3)
        source_solutions = [(optimizer.ask(), 0.0) for _ in range(100)]
        ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions)

        self.assertEqual(ws_mean.size, 10)


@pytest.mark.parametrize("gamma", [0.5, 1.0])
@pytest.mark.parametrize("offset", [(0, 0), (2**28, 0), (0, -(2**28)), (2**28, -(2**28))])
def test_warm_start_covariance_after_translation(gamma, offset):
    offset = np.array(offset, dtype=float)
    points = np.array([[-0.5, -0.25], [-0.25, -0.5], [0.25, 0.5], [0.5, 0.25]])
    solutions = [(x + offset, 0.0) for x in np.tile(points, (2, 1))]
    mean, sigma, cov = get_warm_start_mgd(solutions, gamma=gamma)
    expected = np.array([[0.15625, 0.125], [0.125, 0.15625]]) + 0.1**2 * np.eye(2)
    np.testing.assert_array_equal(mean, offset)
    np.testing.assert_allclose(sigma**2 * cov, expected, rtol=1e-14, atol=0)
    optimizer = CMA(mean=mean, sigma=sigma, cov=cov, seed=0)
    assert np.isfinite(optimizer.ask()).all()
