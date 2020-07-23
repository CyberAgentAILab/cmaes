import random

import numpy as np
from unittest import TestCase
from cmaes import CMA


class TestTerminationCriterion(TestCase):
    def test_stop_if_objective_values_are_not_changed(self):
        optimizer = CMA(mean=np.zeros(2), sigma=1.3)
        popsize = optimizer.population_size
        rng = np.random.RandomState(seed=1)

        for i in range(optimizer._funhist_term):
            optimizer.tell([(rng.randn(2), 0.01) for _ in range(popsize)])
            self.assertFalse(optimizer.should_stop())

        optimizer.tell([(rng.randn(2), 0.01) for _ in range(popsize)])
        self.assertTrue(optimizer.should_stop())

    def test_stop_if_detect_divergent_behavior(self):
        optimizer = CMA(mean=np.zeros(2), sigma=1e-8)
        popsize = optimizer.population_size
        rng = random.Random(1)
        nd_rng = np.random.RandomState(1)

        solutions = [(nd_rng.randn(2), 1000 * rng.random()) for _ in range(popsize)]
        optimizer.tell(solutions)
        self.assertTrue(optimizer.should_stop())
