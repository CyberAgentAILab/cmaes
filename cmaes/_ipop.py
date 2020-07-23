import math
import numpy as np

from ._cma import CMA

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


class IPopCMA:
    """The restart CMA-ES with increasing population (IPOP-CMA-ES).
    This class provides ask-and-tell interface which is the same with
    `CMA` class.

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        inc_popsize:
            Multiplier for increasing population size before each restart.
    """

    def __init__(
        self,
        # Arguments for CMA class.
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        popsize: Optional[int] = None,
        # Arguments for IPOP-CMA-ES.
        inc_popsize: int = 2,
    ):
        self._cma_opts = {
            "mean": mean,
            "sigma": sigma,
            "bounds": bounds,
            "n_max_resampling": n_max_resampling,
            "seed": seed,
        }  # type: Dict[str, Any]

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if popsize is None:
            popsize = 4 + math.floor(3 * math.log(n_dim))
        assert popsize > 0, "popsize must be non-zero positive value."

        self._popsize = popsize
        self._n_restarts = 0
        self._inc_popsize = inc_popsize
        self._cma = CMA(popsize=popsize, **self._cma_opts)

        # termination criterion: tolerance in x-changes [1e-17, ~1e-3]
        self._tolx = 1e-11
        self._tolxup = 1e4

        # termination criterion: tolerance in function value [1e-17, ~1e-3]
        # pycma's default is 1e-11 though paper recommends 1e-12.
        self._tolfun = 1e-11
        self._funcval_generation_term = 10 + math.ceil(30 * n_dim / popsize)
        # stores 'best' and 'worst'
        self._funcval_values = np.empty(self._funcval_generation_term * 2)

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._cma.dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._cma.generation

    def ask(self) -> np.ndarray:
        """Sample a parameter."""
        return self._cma.ask()

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values."""
        self._cma.tell(solutions)

        self._update_objective_values([s[1] for s in solutions])

        if self._should_stop():
            self._restart()

    def _restart(self) -> None:
        self._n_restarts += 1
        self._popsize *= self._inc_popsize
        self._cma = CMA(popsize=self._popsize, **self._cma_opts)

        self._funcval_generation_term = 10 + math.ceil(
            30 * self._cma.dim / self._popsize
        )
        self._funcval_values = np.empty(self._funcval_generation_term * 2)

    def _update_objective_values(self, values: List[float]) -> None:
        # Stores best values and worst values
        # of the last '10 + math.ceil(30 * n / popsize)' generations.
        value_index = 2 * (self.generation % self._funcval_generation_term)
        np_values = np.array(values, dtype=float)
        self._funcval_values[value_index] = np.min(np_values)
        self._funcval_values[value_index + 1] = np.max(np_values)

    def _should_stop(self) -> bool:
        # eigen-decomposition
        C = (self._cma._C + self._cma._C.T) / 2
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(D2)
        dC = np.diag(C)

        # Update cache of B and D
        self._cma._B, self._cma._D = B, D

        # aliases
        sigma = self._cma._sigma
        pc = self._cma._pc

        # Equal objective function values.
        if (
            self.generation > self._funcval_generation_term
            and np.max(self._funcval_values) - np.min(self._funcval_values)
            < self._tolfun
        ):
            return True

        # TolX:
        if np.all(sigma * dC < self._tolx) and np.all(sigma * pc < self._tolx):
            return True

        # TolXUp: This usually indicates a far too small initial sigma,
        # or divergent behavior.
        if sigma * np.max(D) > self._tolxup:
            return True

        # Stop if the condition number of the covariance matrix exceeds 10^14.
        condition_cov = np.max(D) / np.min(D)
        if condition_cov > 1e14:
            return True

        return False
