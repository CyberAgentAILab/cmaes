import sys
import atheris
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies as st

from cmaes import CMA


@given(data=st.data())
def test_cma_tell(data):
    dim = data.draw(st.integers(min_value=2, max_value=100))
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


atheris.Setup(sys.argv, test_cma_tell.hypothesis.fuzz_one_input)
atheris.Fuzz()
