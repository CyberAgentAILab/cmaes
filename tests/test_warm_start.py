import numpy as np
from unittest import TestCase
from cmaes import CMA, get_warm_start_mgd


class TestWarmStartCMA(TestCase):
    def test_dimension(self):
        optimizer = CMA(mean=np.zeros(10), sigma=1.3)
        source_solutions = [(optimizer.ask(), 0.0) for _ in range(100)]
        ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions)

        self.assertEqual(ws_mean.size, 10)
