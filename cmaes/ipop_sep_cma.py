# cmaes/ipop_sep_cma.py
from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple, List, Union

try:
    # Prefer public import to keep user API consistent
    from cmaes import SepCMA
except Exception:
    # Fallback for local development
    from ._sepcma import SepCMA  # type: ignore

ArrayLike = Union[np.ndarray, Sequence[float]]

class IPOPSepCMA:
    """
    IPOP-style restart wrapper around SepCMA.

    - Preserves ask()/tell() interface
    - Doubles population size on each restart
    - Simple, readable stall detection with sensible defaults
    """

    def __init__(
        self,
        mean: ArrayLike,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        population_size: Optional[int] = None,
        seed: Optional[int] = None,
        # --- restart / stall controls ---
        patience: int = 20,                 # gens with no meaningful relative improvement
        tol_rel_improve: float = 1e-12,     # relative improvement threshold
        min_sigma: float = 1e-12,           # restart if step-size collapses
        max_restarts: int = 10,             # global cap
        max_population_size: Optional[int] = None,
        stage_max_generations: Optional[int] = None,   # optional per-stage cap
        # schedule
        inc_popsize: int = 2,
        min_stage_generations: int = 8,
    ):
        self.dim = int(len(mean))
        self._init_mean = np.asarray(mean, dtype=float)
        self._init_sigma = float(sigma)
        self._bounds = bounds
        self._base_pop = int(population_size) if population_size is not None else None
        self._max_pop = int(max_population_size) if max_population_size is not None else None
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.min_stage_generations = int(min_stage_generations)

        # stall detection state
        self.patience = int(patience)
        self.tol_rel_improve = float(tol_rel_improve)
        self.min_sigma = float(min_sigma)
        self.stage_max_generations = stage_max_generations
        self.max_restarts = int(max_restarts)
        self.inc_popsize = int(inc_popsize)

        # tracking
        self.restart_count = 0
        self.best_f = np.inf
        self.best_x: Optional[np.ndarray] = None
        self._no_improve_gens = 0
        self._stage_evals = 0


        # spawn first stage optimizer
        self._spawn(pop_multiplier=1)

    # -------- Public API --------

    @property
    def population_size(self) -> int:
        return int(self._opt.population_size)

    @property
    def generation(self) -> int:
        return int(getattr(self._opt, "generation", 0))

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(getattr(self._opt, "mean", self._init_mean))

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)
        if hasattr(self._opt, "reseed_rng"):
            self._opt.reseed_rng(seed)

    def ask(self) -> np.ndarray:
        return self._opt.ask()
    
    def _account_current_stage_evals(self) -> None:
        # For IPOP we only track total evaluations (no regime concept)
        if not hasattr(self, "_total_n_eval"):
            self._total_n_eval = 0
        self._total_n_eval += self._stage_evals
        self._stage_evals = 0



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
        # global stop: max restarts reached
        return self.restart_count >= self.max_restarts

    # -------- Internals --------

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

    def _restart(self) -> None:
        if self.restart_count >= self.max_restarts:
            return
        self.restart_count += 1
        self._no_improve_gens = 0
        # IPOP: multiply population size by inc_popsize^restarts
        mult = self.inc_popsize ** self.restart_count
        self._spawn(pop_multiplier=mult)

    def _spawn(self, pop_multiplier: int) -> None:
        # Determine base popsize (using SepCMA default if not provided)
        if self._base_pop is None:
            probe = SepCMA(mean=np.zeros(self.dim), sigma=self._init_sigma, seed=self._seed)
            base = int(probe.population_size)
        else:
            base = int(self._base_pop)

        pop = int(base * pop_multiplier)
        if self._max_pop is not None:
            pop = min(pop, self._max_pop)

        # Choose restart mean:
        if self._bounds is not None:
            lb, ub = self._bounds[:, 0], self._bounds[:, 1]
            mean = lb + self._rng.rand(self.dim) * (ub - lb)
        elif self.best_x is not None and np.all(np.isfinite(self.best_x)):
            # small jitter around best-so-far
            mean = self.best_x + self._rng.randn(self.dim) * (0.5 * self._init_sigma)
        else:
            mean = self._init_mean

        self._opt = SepCMA(
            mean=np.asarray(mean, float),
            sigma=self._init_sigma,            # keep a stable stage step-size
            bounds=self._bounds,
            population_size=pop,
            seed=self._seed,
        )
