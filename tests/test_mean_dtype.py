import numpy as np
import pytest

from cmaes import CMA, SepCMA


@pytest.mark.parametrize("optimizer_class", [CMA, SepCMA])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64, np.float32, np.float64])
def test_mean_dtype_does_not_change_optimization(optimizer_class, dtype):
    mean = np.array([1, 2, 3], dtype=dtype)
    original = mean.copy()
    optimizer = optimizer_class(mean=mean, sigma=1.3, seed=42)
    reference = optimizer_class(mean=mean.astype(float), sigma=1.3, seed=42)

    for _ in range(5):
        solutions = []
        reference_solutions = []
        for _ in range(optimizer.population_size):
            point = optimizer.ask()
            reference_point = reference.ask()
            np.testing.assert_allclose(point, reference_point, rtol=1e-12, atol=1e-12)
            solutions.append((point, float(point @ point)))
            reference_solutions.append((reference_point, float(reference_point @ reference_point)))
        optimizer.tell(solutions)
        reference.tell(reference_solutions)
        np.testing.assert_allclose(optimizer.mean, reference.mean, rtol=1e-12, atol=1e-12)
        assert np.isfinite(optimizer.mean).all()

    np.testing.assert_array_equal(mean, original)
    assert not np.shares_memory(optimizer.mean, mean)
