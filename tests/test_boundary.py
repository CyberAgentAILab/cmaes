import numpy as np
from unittest import TestCase
from cmaes import CMA, SepCMA


CMA_CLASSES = [CMA, SepCMA]


class TestCMABoundary(TestCase):
    def test_valid_dimension(self):
        for CmaClass in CMA_CLASSES:
            with self.subTest(f"Class: {CmaClass.__name__}"):
                CmaClass(
                    mean=np.zeros(2), sigma=1.3, bounds=np.array([[-10, 10], [-10, 10]])
                )

    def test_invalid_dimension(self):
        for CmaClass in CMA_CLASSES:
            with self.subTest(f"Class: {CmaClass.__name__}"):
                with self.assertRaises(AssertionError):
                    CmaClass(mean=np.zeros(2), sigma=1.3, bounds=np.array([-10, 10]))

    def test_mean_located_out_of_bounds(self):
        mean = np.zeros(5)
        bounds = np.empty(shape=(5, 2))
        bounds[:, 0], bounds[:, 1] = 1.0, 5.0
        for CmaClass in CMA_CLASSES:
            with self.subTest(f"Class: {CmaClass.__name__}"):
                with self.assertRaises(AssertionError):
                    CmaClass(mean=mean, sigma=1.3, bounds=bounds)

    def test_set_valid_bounds(self):
        for CmaClass in CMA_CLASSES:
            with self.subTest(f"Class: {CmaClass.__name__}"):
                optimizer = CmaClass(mean=np.zeros(2), sigma=1.3)
                optimizer.set_bounds(bounds=np.array([[-10, 10], [-10, 10]]))

    def test_set_invalid_bounds(self):
        for CmaClass in CMA_CLASSES:
            with self.subTest(f"Class: {CmaClass.__name__}"):
                optimizer = CmaClass(mean=np.zeros(2), sigma=1.3)
                with self.assertRaises(AssertionError):
                    optimizer.set_bounds(bounds=np.array([-10, 10]))

    def test_set_bounds_which_does_not_contain_mean(self):
        for CmaClass in CMA_CLASSES:
            with self.subTest(f"Class: {CmaClass.__name__}"):
                optimizer = CmaClass(mean=np.zeros(2), sigma=1.3)
                bounds = np.empty(shape=(5, 2))
                bounds[:, 0], bounds[:, 1] = 1.0, 5.0
                with self.assertRaises(AssertionError):
                    optimizer.set_bounds(bounds)
