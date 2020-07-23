import numpy as np
from unittest import TestCase
from cmaes._cma import _decompress_symmetric, _compress_symmetric


class TestCompressSymmetric(TestCase):
    def test_compress_symmetric_odd(self):
        sym2d = np.array([[1, 2], [2, 3]])
        actual = _compress_symmetric(sym2d)
        expected = np.array([1, 2, 3])
        self.assertTrue(np.all(np.equal(actual, expected)))

    def test_compress_symmetric_even(self):
        sym2d = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        actual = _compress_symmetric(sym2d)
        expected = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.all(np.equal(actual, expected)))

    def test_decompress_symmetric_odd(self):
        sym1d = np.array([1, 2, 3])
        actual = _decompress_symmetric(sym1d)
        expected = np.array([[1, 2], [2, 3]])
        self.assertTrue(np.all(np.equal(actual, expected)))

    def test_decompress_symmetric_even(self):
        sym1d = np.array([1, 2, 3, 4, 5, 6])
        actual = _decompress_symmetric(sym1d)
        expected = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        self.assertTrue(np.all(np.equal(actual, expected)))
