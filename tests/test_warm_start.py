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


@pytest.mark.parametrize("dim", [2, 10, 200, 256])
@pytest.mark.parametrize("alpha", [0.1, 10.0])
def test_warm_start_scale_in_high_dimensions(dim, alpha):
    solutions = [(np.zeros(dim), 0.0) for _ in range(10)]
    mean, sigma, cov = get_warm_start_mgd(solutions, alpha=alpha)
    np.testing.assert_allclose(sigma, alpha, rtol=1e-12)
    np.testing.assert_allclose(cov, np.eye(dim), rtol=1e-12, atol=0)
    optimizer = CMA(mean=mean, sigma=sigma, cov=cov, seed=0)
    assert np.isfinite(optimizer.ask()).all()


def test_warm_start_scale_with_subnormal_determinant():
    alpha = 1e-80
    solutions = [(np.zeros(2), 0.0) for _ in range(10)]
    _, sigma, cov = get_warm_start_mgd(solutions, alpha=alpha)
    np.testing.assert_allclose(sigma, alpha, rtol=1e-12, atol=0)
    np.testing.assert_allclose(cov, np.eye(2), rtol=1e-12, atol=0)
