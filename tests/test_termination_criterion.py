import numpy as np

from unittest import TestCase
from cmaes import CMA


class TestTerminationCriterion(TestCase):
    def test_stop_if_objective_values_are_not_changed(self):
        optimizer = CMA(mean=np.zeros(2), sigma=1.3)
        popsize = optimizer.population_size
        rng = np.random.RandomState(seed=1)

        for i in range(optimizer._funhist_term + 1):
            self.assertFalse(optimizer.should_stop())
            optimizer.tell([(rng.randn(2), 0.01) for _ in range(popsize)])

        self.assertTrue(optimizer.should_stop())

    def test_stop_if_detect_divergent_behavior(self):
        optimizer = CMA(mean=np.zeros(2), sigma=1e-4)
        popsize = optimizer.population_size
        nd_rng = np.random.RandomState(1)

        solutions = [(100 * nd_rng.randn(2), 0.01) for _ in range(popsize)]
        optimizer.tell(solutions)
        self.assertTrue(optimizer.should_stop())
