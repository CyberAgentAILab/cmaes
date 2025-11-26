# cmaes/bipop_full_cma.py
from __future__ import annotations
import numpy as np, math
from typing import Optional, Sequence, Tuple, List, Union
from .cma import CMA

ArrayLike = Union[np.ndarray, Sequence[float]]

class BIPOPFullCMA:
    """BIPOP wrapper for full CMA-ES (ask/tell compatible)."""
    def __init__(self, mean: ArrayLike, sigma: float, bounds: Optional[np.ndarray]=None,
                 population_size: Optional[int]=None, seed: Optional[int]=None,
                 inc_popsize: int=2, patience: int=20, tol_rel_improve: float=1e-8,
                 min_sigma: float=1e-8, stage_max_generations: Optional[int]=None,
                 max_restarts: int=50, max_population_size: Optional[int]=None):
        self.dim = int(len(mean))
        self._init_mean = np.asarray(mean, float)
        self._init_sigma = float(sigma)
        self._bounds = bounds
        self._base_pop = int(population_size) if population_size is not None else None
        self._max_pop = int(max_population_size) if max_population_size is not None else None
        self.inc_popsize = int(inc_popsize)
        self.patience = int(patience)
        self.tol_rel_improve = float(tol_rel_improve)
        self.min_sigma = float(min_sigma)
        self.stage_max_generations = stage_max_generations
        self.max_restarts = int(max_restarts)
        self._seed = int(seed) if seed is not None else None
        self._rng = np.random.RandomState(self._seed)
        self.restart_count = 0
        self.best_f = np.inf
        self.best_x: Optional[np.ndarray] = None
        self._no_improve_gens = 0
        self._stage_evals = 0
        self._regime = "small"
        self._small_n_eval = 0
        self._large_n_eval = 0
        self._spawn_initial()

    @property
    def population_size(self) -> int: return int(self._opt.population_size)
    @property
    def generation(self) -> int: return int(getattr(self._opt, "generation", 0))
    @property
    def regime(self) -> str: return self._regime

    def _spawn_initial(self) -> None:
        if self._base_pop is None:
            probe = CMA(mean=np.zeros(self.dim), sigma=self._init_sigma, seed=self._seed)
            self._pop0 = int(probe.population_size)
        else:
            self._pop0 = int(self._base_pop)
        pop, sigma = self._small_pop_and_sigma()
        self._opt = CMA(mean=self._restart_mean(), sigma=sigma, bounds=self._bounds, seed=self._seed, population_size=pop)

    def _restart(self) -> None:
        go_small = self._small_n_eval < self._large_n_eval
        if go_small:
            self._regime = "small"; pop, sigma = self._small_pop_and_sigma()
        else:
            self._regime = "large"; self.restart_count += 1; pop, sigma = self._large_pop_and_sigma()
        self._opt = CMA(mean=self._restart_mean(), sigma=sigma, bounds=self._bounds, seed=self._seed, population_size=pop)
        self._no_improve_gens = 0; self._stage_evals = 0

    def _large_pop_and_sigma(self) -> Tuple[int,float]:
        pop = int(self._pop0 * (self.inc_popsize ** self.restart_count))
        if self._max_pop is not None: pop = min(pop, self._max_pop)
        return pop, self._init_sigma

    def _small_pop_and_sigma(self) -> Tuple[int,float]:
        U = self._rng.uniform()
        mult = (self.inc_popsize ** max(self.restart_count, 1)) ** (U * U)
        pop = int(max(2, math.floor(self._pop0 * mult)))
        if self._max_pop is not None: pop = min(pop, self._max_pop)
        sigma = self._init_sigma * (10 ** (-2.0 * U))
        return pop, sigma

    def _restart_mean(self) -> np.ndarray:
        if self._bounds is not None:
            lb, ub = self._bounds[:,0], self._bounds[:,1]
            return (lb + self._rng.rand(self.dim)*(ub-lb)).astype(float)
        if self.best_x is not None and np.all(np.isfinite(self.best_x)):
            return (self.best_x + self._rng.randn(self.dim) * (0.5 * self._init_sigma)).astype(float)
        return self._init_mean

    def _should_restart(self) -> bool:
        if getattr(self._opt, "should_stop", lambda: False)() and self.generation >= 8:
            return True
        if self._no_improve_gens >= self.patience: return True
        if self.stage_max_generations is not None and self.generation >= self.stage_max_generations: return True
        if float(getattr(self._opt, "sigma", self._init_sigma)) < self.min_sigma: return True
        return False

    def ask(self) -> np.ndarray: return self._opt.ask()

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        bx, bf = min(solutions, key=lambda t: t[1])
        if bf < self.best_f:
            prev = self.best_f
            self.best_f = float(bf); self.best_x = np.asarray(bx, float)
            rel = (prev - self.best_f) / (abs(prev) + 1e-12) if np.isfinite(prev) else np.inf
            self._no_improve_gens = 0 if rel > self.tol_rel_improve else self._no_improve_gens + 1
        else:
            self._no_improve_gens += 1
        self._stage_evals += len(solutions)
        self._opt.tell(solutions)
        if self._should_restart():
            if self._regime == "small": self._small_n_eval += self._stage_evals
            else: self._large_n_eval += self._stage_evals
            self._stage_evals = 0
            self._restart()

    def should_stop(self) -> bool:
        return self.restart_count >= self.max_restarts
