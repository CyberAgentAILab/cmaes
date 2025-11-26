# cmaes/bipop_sep_cma.py
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Sequence, Tuple, List, Union

try:
    from cmaes import SepCMA
except Exception:
    from ._sepcma import SepCMA  # type: ignore

ArrayLike = Union[np.ndarray, Sequence[float]]

class BIPOPSepCMA:
    """
    BIPOP-style restart wrapper around SepCMA (ask/tell compatible).

    - Alternates between 'small' and 'large' population regimes
    - Chooses next regime by comparing cumulated evals spent in each regime
      (allocate more to the under-spent regime)
    - Large regime: popsize = pop0 * (inc_popsize ** n_restarts), sigma = sigma0
    - Small regime: popsize ~ pop0 * (inc_popsize ** n_restarts) ** (U^2), sigma = sigma0 * 10^(-2U),
      with U ~ Uniform(0,1)
    - Optional stall detection (patience / sigma collapse / stage gen cap)
    """

    def __init__(
        self,
        mean: ArrayLike,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        population_size: Optional[int] = None,
        seed: Optional[int] = None,
        # BIPOP schedule
        inc_popsize: int = 2,
        # Stall detection (optional; inner SepCMA also has its stops)
        patience: int = 20,
        tol_rel_improve: float = 1e-12,
        min_sigma: float = 1e-12,
        stage_max_generations: Optional[int] = None,
        max_restarts: int = 50,
        max_population_size: Optional[int] = None,
        min_stage_generations: int = 8,
    ):
        
        self.dim = int(len(mean))
        self._init_mean = np.asarray(mean, float)
        self._init_sigma = float(sigma)
        self._bounds = bounds
        self._base_pop = int(population_size) if population_size is not None else None
        self._max_pop = int(max_population_size) if max_population_size is not None else None
        self.min_stage_generations = int(min_stage_generations)

        self._seed = int(seed) if seed is not None else None
        self._rng = np.random.RandomState(self._seed)

        # schedule / criteria
        self.inc_popsize = int(inc_popsize)
        self.patience = int(patience)
        self.tol_rel_improve = float(tol_rel_improve)
        self.min_sigma = float(min_sigma)
        self.stage_max_generations = stage_max_generations
        self.max_restarts = int(max_restarts)

        # bookkeeping
        self.restart_count = 0           # counts only the "large" regime restarts (as in BIPOP)
        self.best_f = np.inf
        self.best_x: Optional[np.ndarray] = None
        self._no_improve_gens = 0

        # regime state
        self._regime = "small"           # initial regime is considered "small"
        self._small_n_eval = 0
        self._large_n_eval = 0
        self._stage_evals = 0

        # initial spawn
        self._spawn_initial()

    # ------------- public API -------------
    @property
    def population_size(self) -> int:
        return int(self._opt.population_size)

    @property
    def generation(self) -> int:
        return int(getattr(self._opt, "generation", 0))

    def ask(self) -> np.ndarray:
        return self._opt.ask()

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        # track best & stall
        bx, bf = min(solutions, key=lambda t: t[1])
        if bf < self.best_f:
            prev = self.best_f
            self.best_f = float(bf)
            self.best_x = np.asarray(bx, float)
            rel = (prev - self.best_f) / (abs(prev) + 1e-12) if np.isfinite(prev) else np.inf
            self._no_improve_gens = 0 if rel > self.tol_rel_improve else self._no_improve_gens + 1
        else:
            self._no_improve_gens += 1

        # ✅ ADD THIS LINE right *before* self._opt.tell(solutions)
        self._stage_evals += len(solutions)

        # pass solutions to inner optimizer
        self._opt.tell(solutions)

        # check for restart
        if self._should_restart():
            self._account_current_stage_evals()
            self._restart()


    def should_stop(self) -> bool:
        # stop after many "large" restarts
        return self.restart_count >= self.max_restarts

    # ------------- internals -------------
    def _should_restart(self) -> bool:
        # avoid instant restarts right after spawn
        if self.generation < self.min_stage_generations:
            return False

        # our criteria first
        if self._no_improve_gens >= self.patience:
            return True
        if self.stage_max_generations is not None and self.generation >= self.stage_max_generations:
            return True
        if float(getattr(self.opt if hasattr(self, "opt") else self._opt, "sigma", self._init_sigma)) < self.min_sigma:
            return True

        # then inner stop AFTER cooldown
        if hasattr(self.opt if hasattr(self, "opt") else self._opt, "should_stop") and \
        (self.opt if hasattr(self, "opt") else self._opt).should_stop():
            return True

        return False

    def _account_current_stage_evals(self) -> None:
        if self._regime == "small":
            self._small_n_eval += self._stage_evals
        else:
            self._large_n_eval += self._stage_evals
        self._stage_evals = 0


    def _spawn_initial(self) -> None:
        # determine base popsize via SepCMA default if not provided
        if self._base_pop is None:
            probe = SepCMA(mean=np.zeros(self.dim), sigma=self._init_sigma, seed=self._seed)
            self._pop0 = int(probe.population_size)
        else:
            self._pop0 = int(self._base_pop)
        # initial regime is "small" with randomized small-pop parameters
        pop, sigma = self._small_pop_and_sigma()
        mean = self._draw_restart_mean()
        self._opt = SepCMA(mean=mean, sigma=sigma, bounds=self._bounds, seed=self._seed, population_size=pop)
        self._no_improve_gens = 0

    def _restart(self) -> None:
        # choose next regime:
        # if small budget spent < large budget spent -> go small; else go large (+1 restart count)
        go_small = self._small_n_eval < self._large_n_eval

        if go_small:
            self._regime = "small"
            pop, sigma = self._small_pop_and_sigma()
        else:
            self._regime = "large"
            self.restart_count += 1
            pop, sigma = self._large_pop_and_sigma()

        mean = self._draw_restart_mean()
        self._opt = SepCMA(mean=mean, sigma=sigma, bounds=self._bounds, seed=self._seed, population_size=pop)
        self._no_improve_gens = 0

    # ---- regime helpers ----
    def _large_pop_and_sigma(self) -> Tuple[int, float]:
        # pop = pop0 * (inc^n_restarts)
        pop = int(self._pop0 * (self.inc_popsize ** self.restart_count))
        if self._max_pop is not None:
            pop = min(pop, self._max_pop)
        sigma = self._init_sigma
        return pop, sigma

    def _small_pop_and_sigma(self) -> Tuple[int, float]:
        # pop = floor(pop0 * (inc^n_restarts) ** (U^2))
        U = self._rng.uniform()
        mult = (self.inc_popsize ** max(self.restart_count, 1)) ** (U * U)
        pop = int(math.floor(self._pop0 * mult))
        pop = max(pop, 2)  # guard
        if self._max_pop is not None:
            pop = min(pop, self._max_pop)
        # sigma = sigma0 * 10^(-2U)
        sigma = self._init_sigma * (10 ** (-2.0 * U))
        return pop, sigma

    def _draw_restart_mean(self) -> np.ndarray:
        if self._bounds is not None:
            lb, ub = self._bounds[:, 0], self._bounds[:, 1]
            return (lb + self._rng.rand(self.dim) * (ub - lb)).astype(float)
        if self.best_x is not None and np.all(np.isfinite(self.best_x)):
            return (self.best_x + self._rng.randn(self.dim) * (0.5 * self._init_sigma)).astype(float)
        return self._init_mean.astype(float)
    
    @property
    def regime(self) -> str:
        return self._regime

