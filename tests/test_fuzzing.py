import hypothesis.extra.numpy as npst
import unittest
from hypothesis import given, strategies as st

from cmaes import CMA, SepCMA


class TestFuzzing(unittest.TestCase):
    @given(
        data=st.data(),
    )
    def test_cma_tell(self, data):
        dim = data.draw(st.integers(min_value=1, max_value=100))
        mean = data.draw(npst.arrays(dtype=float, shape=dim))
        sigma = data.draw(st.floats(min_value=1e-16))
        n_iterations = data.draw(st.integers(min_value=1))
        try:
            optimizer = CMA(mean, sigma)
        except AssertionError:
            return
        popsize = optimizer.population_size
        for _ in range(n_iterations):
            tell_solutions = data.draw(
                st.lists(
                    st.tuples(npst.arrays(dtype=float, shape=dim), st.floats()),
                    min_size=popsize,
                    max_size=popsize,
                )
            )
            optimizer.ask()
            try:
                optimizer.tell(tell_solutions)
            except AssertionError:
                return
            optimizer.ask()

    @given(
        data=st.data(),
    )
    def test_sepcma_tell(self, data):
        dim = data.draw(st.integers(min_value=2, max_value=100))
        mean = data.draw(npst.arrays(dtype=float, shape=dim))
        sigma = data.draw(st.floats(min_value=1e-16))
        n_iterations = data.draw(st.integers(min_value=1))
        try:
            optimizer = SepCMA(mean, sigma)
        except AssertionError:
            return
        popsize = optimizer.population_size
        for _ in range(n_iterations):
            tell_solutions = data.draw(
                st.lists(
                    st.tuples(npst.arrays(dtype=float, shape=dim), st.floats()),
                    min_size=popsize,
                    max_size=popsize,
                )
            )
            optimizer.ask()
            try:
                optimizer.tell(tell_solutions)
            except AssertionError:
                return
            optimizer.ask()
