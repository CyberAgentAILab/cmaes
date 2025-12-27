from __future__ import annotations

import numpy as np
from collections import deque

from dataclasses import dataclass, field

from typing import overload
from typing import List, Deque, Sequence, Union, Tuple, Iterator, Optional

from ._catcmawm import CatCMAwM
from ._uhvi_archiving import UHVIArchive2D


class AskQueueEmpty(RuntimeError):
    """No askable solutions are available right now."""


class COMOCatCMAwM:
    """COMO-CatCMA with Margin stochastic multi-objective optimizer class with
    ask-and-tell interface.

    Example:

        .. code::

            from cmaes import COMOCatCMAwM

            def DSInt(x, z):
                f1 = sum(x ** 2) + sum(z ** 2)
                f2 = sum((x - 1) ** 2) + sum((z - 1) ** 2)
                return [f1, f2]

            X = [[-1, 2], [-1, 2]]
            Z = [[0, 0.5, 1], [-0.5, 0, 0.5, 1, 1.5]]

            optimizer = COMOCatCMAwM(x_space=X, z_space=Z)

            evals = 0
            while evals < 1000:
                solutions = []
                for sol in optimizer.ask_iter():
                    value = DSInt(sol.x, sol.z)
                    evals += 1
                    solutions.append((sol, value))
                optimizer.tell(solutions)
                print(evals, optimizer.incumbent_objectives)

    Args:

        x_space:
            The search space for continuous variables.
            Provide as a 2-dimensional sequence (e.g., a list of lists),
            where each row is [lower_bound, upper_bound] for a continuous variable.
            If there are no continuous variables, this parameter can be omitted.
            Example: [[-3.0, 3.0], [0.0, 5.0], [-np.inf, np.inf]]

        z_space:
            The set of possible values for each integer variable.
            Provide as a list of lists, where each inner list contains the valid
            (sorted) integer or discretized values for that variable.
            If there are no integer variables, this parameter can be omitted.
            Example: [[-2, -1, 0, 1, 2], [0.01, 0.1, 1]]
            Note: For binary variables (i.e., variables that can only take two distinct values),
            it is generally recommended to use the categorical variable representation via
            `c_space` rather than treating them as integer variables.

        c_space:
            The shape of the categorical variables' domain.
            Provide as a 1-dimensional sequence (e.g., a list)
            where each element specifies the number of categories (integer > 1)
            for each categorical variable.
            If there are no categorical variables, this parameter can be omitted.
            Example: [3, 3, 2, 10]
            Note: Binary variables (with only two possible values) should be represented as
            categorical variables here, rather than as integer variables in `z_space`.

        reference_point:
            A fixed reference point for the objective space (optional).

        kernels:
            Initial CatCMAwM optimizers to run in parallel (optional).

        kernel_size:
            Number of internal CatCMAwM optimizers when 'kernels' is not supplied
            (optional, default: 10).

        seed:
            A seed number (optional).
    """

    # Currently limited to 2 objectives due to 2D UHVI constraints; expansion is planned.
    _MAX_OBJECTIVES: int = 2

    @dataclass(frozen=True)
    class Solution:
        x: Optional[np.ndarray] = None  # continuous variable
        z: Optional[np.ndarray] = None  # integer variable
        c: Optional[np.ndarray] = None  # categorical variable
        _v_raw: Optional[np.ndarray] = field(default=None, repr=False)  # internal use
        _kernel_id: Optional[int] = field(default=None, repr=False)  # internal use
        _incumbent_id: Optional[int] = field(default=None, repr=False)  # internal use

    def __init__(
        self,
        x_space: Optional[Sequence[Sequence[float]]] = None,
        z_space: Optional[Sequence[Sequence[Union[int, float]]]] = None,
        c_space: Optional[Sequence[int]] = None,
        reference_point: Optional[Sequence[float]] = None,
        kernels: Optional[Sequence[CatCMAwM]] = None,
        kernel_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # Determine space sizes
        self._Nco = len(x_space) if x_space is not None else 0
        self._Nin = len(z_space) if z_space is not None else 0
        self._Nca = len(c_space) if c_space is not None else 0
        self._Nmi = self._Nco + self._Nin

        if self._Nmi + self._Nca <= 0:
            raise ValueError("The total number of dimensions must be positive.")

        self._use_continuous = self._Nco > 0
        self._use_integer = self._Nin > 0
        self._use_gaussian = self._Nmi > 0
        self._use_categorical = self._Nca > 0

        # global seed
        self._rng = np.random.RandomState(seed)

        # initialization of kernels
        if kernels is None:
            if kernel_size is None:
                kernel_size = 10
            if kernel_size <= 0:
                raise ValueError("kernel_size must be positive.")
            kernels = self._init_kernels(kernel_size, x_space, z_space, c_space)
        self._kernel_size = len(kernels)
        if self._kernel_size <= 0:
            raise ValueError("The number of kernels must be positive.")
        self._kernels = kernels

        # Seed the global RNG and all internal kernels
        if seed is not None:
            self.reseed_rng(seed)

        # generation number
        self._g: int = 0

        # Randomize kernel update sequence
        self._kernel_pi: List[int] = (self._rng.permutation(self._kernel_size)).tolist()
        self._kernel_index: int = 0

        # Set up incumbent solutions and evaluation status
        self._incumbents_sol = np.array(
            [self.calc_incumbent_solution(i) for i in range(self._kernel_size)]
        )
        self._incumbents_obj: np.ndarray | None = None
        self._incumbents_mask: int = 0

        # Initialize objective dimension and user-provided reference point
        self._n_obj: int | None = None
        self._fixed_ref_point: np.ndarray | None = None
        if reference_point is not None:
            self._fixed_ref_point = self._coerce_objective_space_point(reference_point)

        # for dynamic reference point
        self._ideal_point: np.ndarray | None = None
        self._nadir_point: np.ndarray | None = None
        self._dynamic_ref_scale: float = 0.1
        self._dynamic_ref_min_range: float = 1e-7

        # queue for pending candidate solutions
        self._ask_queue: Deque[COMOCatCMAwM.Solution] = deque()

        # per-kernel buffers for tell observations
        self._tell_lists: List[List[Tuple[COMOCatCMAwM.Solution, Sequence[float]]]] = [
            [] for _ in range(self._kernel_size)
        ]

        # Enqueue incumbent solutions of all kernels for evaluation
        for i in range(self._kernel_size):
            (
                self._enqueue_kernel_asks
                if i == self.current_kernel_id
                else self._enqueue_kernel_incumbent
            )(i)

    def _init_kernels(
        self,
        kernel_size: int,
        x_space: Optional[Sequence[Sequence[float]]] = None,
        z_space: Optional[Sequence[Sequence[Union[int, float]]]] = None,
        c_space: Optional[Sequence[int]] = None,
    ) -> Sequence[CatCMAwM]:
        if not self._use_gaussian:
            kernels = [
                CatCMAwM(
                    x_space=x_space,
                    z_space=z_space,
                    c_space=c_space,
                    # CatCMAwM initializes cat_param uniformly
                )
                for _ in range(kernel_size)
            ]
            return kernels

        if x_space is not None:
            x_range = np.asarray(x_space, dtype=float)
            if x_range.ndim != 2 or x_range.shape[1] != 2:
                raise ValueError(
                    f"x_space must be a two-dimensional array with shape (n, 2), "
                    f"but got shape {x_range.shape}."
                )
            invalid = np.where(x_range[:, 0] >= x_range[:, 1])[0]
            if invalid.size > 0:
                i = invalid[0]
                lb, ub = x_range[i, 0], x_range[i, 1]
                raise ValueError(
                    f"Lower bound must be less than upper bound at index {i}: {lb} >= {ub}."
                )

        if z_space is not None:
            z_range = np.empty((len(z_space), 2), dtype=float)
            for i, row in enumerate(z_space):
                if len(row) < 2:
                    raise ValueError(
                        f"z_space must be a sequence of arrays with length >= 2. "
                        f"Found length {len(row)} at index {i}: {row}."
                    )
                s = set(row)
                if len(s) < len(row):
                    raise ValueError(
                        f"Elements in each array of z_space must be unique. "
                        f"Found duplicate at index {i}: {row}."
                    )
                z_range[i, 0] = min(s)
                z_range[i, 1] = max(s)

        # bounds for the mixed continuous and integer space
        if self._use_continuous and self._use_integer:
            lower_x = x_range[:, 0]
            upper_x = x_range[:, 1]
            lower_z = z_range[:, 0]
            upper_z = z_range[:, 1]
            lower_g = np.concatenate([lower_x, lower_z])
            upper_g = np.concatenate([upper_x, upper_z])

        # bounds for the integer space
        if not self._use_continuous and self._use_integer:
            lower_g = z_range[:, 0]
            upper_g = z_range[:, 1]

        # bounds for the continuous space
        if self._use_continuous and not self._use_integer:
            lower_g = x_range[:, 0]
            upper_g = x_range[:, 1]

        # limit to finite range
        neg_inf = np.isneginf(lower_g)
        pos_inf = np.isposinf(upper_g)

        both = neg_inf & pos_inf
        only_low = neg_inf & ~pos_inf
        only_up = pos_inf & ~neg_inf

        lower_g[both] = -1.0
        upper_g[both] = 1.0
        lower_g[only_low] = upper_g[only_low] - 2.0
        upper_g[only_up] = lower_g[only_up] + 2.0

        # Uniform sampling for initial mean vectors
        init_means = self._rng.uniform(lower_g, upper_g, size=(kernel_size, self._Nmi))

        # Set initial standard deviation to
        # width / 6 (continuous)
        # width / 5 (integer)
        width = upper_g - lower_g
        init_cov = np.diag(width / np.where(np.arange(self._Nmi) < self._Nco, 6, 5))

        kernels = [
            CatCMAwM(
                x_space=x_space,
                z_space=z_space,
                c_space=c_space,
                mean=init_mean,
                cov=init_cov,
                # CatCMAwM initializes cat_param uniformly
            )
            for init_mean in init_means
        ]
        return kernels

    @property
    def generation(self) -> int:
        """Generation number incremented after all kernels are updated once."""
        return self._g

    @property
    def kernel_size(self) -> int:
        """Number of kernels."""
        return self._kernel_size

    @property
    def ask_queue_size(self) -> int:
        """Number of solutions currently available to ask."""
        return len(self._ask_queue)

    @property
    def current_kernel_id(self) -> int:
        """ID of the currently active kernel."""
        return self._kernel_pi[self._kernel_index]

    @property
    def kernels(self) -> Sequence[CatCMAwM]:
        """List of all internal kernel optimizers."""
        return self._kernels

    @property
    def current_kernel(self) -> CatCMAwM:
        """The currently active kernel instance."""
        return self._kernels[self.current_kernel_id]

    @property
    def incumbent_solutions(self) -> np.ndarray:
        """Incumbent solutions for all kernels."""
        return self._incumbents_sol

    @property
    def incumbent_objectives(self) -> np.ndarray | None:
        """Objective values of the incumbent solutions."""
        return self._incumbents_obj

    def reseed_rng(self, seed: int) -> None:
        """Set the global seed and reseed the kernels."""
        self._rng.seed(seed)
        subseeds = self._rng.randint(0, 2**32, size=self._kernel_size, dtype=np.uint32)
        for kernel, s in zip(self._kernels, subseeds):
            kernel.reseed_rng(int(s))

    def calc_incumbent_solution(self, kernel_id: int) -> CatCMAwM.Solution:
        """Compute a kernel's incumbent solution."""
        if not (0 <= kernel_id < self._kernel_size):
            raise IndexError(
                f"kernel_id must be in [0, {self._kernel_size}), got {kernel_id}."
            )

        ker = self._kernels[kernel_id]
        x = None
        z = None
        c = None
        v_raw = None
        if ker._use_gaussian:
            v_raw = ker._mean
            if ker._use_continuous:
                x = ker._mean[ker._continuous_idx]
            if ker._use_integer:
                z = ker._discretization(ker._mean[ker._discrete_idx])
        if ker._use_categorical:
            row_max = ker._q.max(axis=1, keepdims=True)
            tie_mask = ker._q == row_max
            argmax_index = np.where(
                tie_mask, ker._rng.random(ker._q.shape), -np.inf
            ).argmax(axis=1)
            c = np.zeros_like(ker._q, dtype=bool)
            c[np.arange(ker._q.shape[0]), argmax_index] = True
        return CatCMAwM.Solution(x, z, c, v_raw)

    def _coerce_objective_space_point(self, point: Sequence[float]) -> np.ndarray:
        """
        Coerce and validate a point in objective space.
        Accepts array-like input and returns a 1-D float numpy array.
        """
        # Coerce array-like input (e.g., list/tuple) to a float numpy array
        p = np.asarray(point, dtype=float)

        # Objective-space points are represented as 1-D vectors of length = number of objectives
        if p.ndim != 1:
            raise ValueError(f"Objective-space point must be 1-D, got shape={p.shape}.")

        p_n_obj: int = p.shape[0]

        # Infer the objective dimension from the first observed point
        # Subsequent points must be consistent with the configured dimension
        if self._n_obj is None:
            self._n_obj = p_n_obj
            # Initialize objective tracking array once the dimension is known
            self._incumbents_obj = np.full(
                (self._kernel_size, self._n_obj), np.nan, dtype=float
            )
        else:
            if self._n_obj != p_n_obj:
                raise ValueError(
                    f"Inconsistent objective dimension: expected {self._n_obj}, got {p_n_obj}."
                )

        # This optimizer is multi-objective; single-objective (dim=1) is not supported here
        if p_n_obj <= 1:
            raise ValueError(f"Objective dimension must be >= 2, got {p_n_obj}.")

        # Guardrail for future extensions: the supported objective dimension may be increased later
        if p_n_obj > self._MAX_OBJECTIVES:
            raise NotImplementedError(
                f"Objective dimension {p_n_obj} is not supported yet "
                f"(currently supports up to {self._MAX_OBJECTIVES})."
            )

        # Return the coerced numpy representation
        return p

    def _increment_kernel_index(self) -> None:
        self._kernel_index += 1
        if self._kernel_index == self._kernel_size:
            self._g += 1
            self._kernel_index = 0
            self._kernel_pi = (self._rng.permutation(self._kernel_size)).tolist()

    def _update_ideal_nadir(self, obj: np.ndarray) -> None:
        if self._ideal_point is None or self._nadir_point is None:
            self._ideal_point = obj.copy()
            self._nadir_point = obj.copy()
        else:
            self._ideal_point = np.minimum(self._ideal_point, obj)
            self._nadir_point = np.maximum(self._nadir_point, obj)

    def _get_reference_point(self) -> np.ndarray:
        """Return the reference point (fixed if provided; otherwise inferred from ideal/nadir).
        Dynamic inference mirrors BoTorch's `infer_reference_point`.
        Default scaling is motivated by Ishibuchi et al. (GECCO 2011).

        BoTorch source:
        https://botorch.readthedocs.io/en/latest/_modules/botorch/utils/multi_objective/hypervolume.html

        Ishibuchi et al. (GECCO 2011):
        Hisao Ishibuchi, Naoya Akedo, and Yusuke Nojima. A many-objective test problem
        for visually examining diversity maintenance behavior in a decision space.
        """
        if self._fixed_ref_point is not None:
            return self._fixed_ref_point

        if self._ideal_point is None or self._nadir_point is None:
            raise RuntimeError(
                "Reference point cannot be inferred yet: no objective values have been observed."
            )

        obj_range = self._nadir_point - self._ideal_point
        obj_range = np.maximum(obj_range, self._dynamic_ref_min_range)

        ref_point = self._nadir_point + self._dynamic_ref_scale * obj_range
        return ref_point

    def _UHVI(self, objective: np.ndarray, kernel_id_to_omit: int) -> float:
        if self._incumbents_obj is None:
            raise ValueError(
                "UHVI unavailable: no incumbent objectives have been recorded yet."
            )
        ref_point = self._get_reference_point()
        # Build incumbents_obj excluding kernel_id_to_omit
        mask = np.ones(self._kernel_size, dtype=bool)
        mask[kernel_id_to_omit] = False
        front = UHVIArchive2D(self._incumbents_obj[mask])
        uhvi_indicator = front.uhvi(objective, ref_point)
        return uhvi_indicator

    def _enqueue_kernel_asks(self, kernel_id: int) -> None:
        """Enqueue all candidate solutions from a specific kernel for evaluation."""
        for _ in range(self._kernels[kernel_id].population_size):
            sol = self._kernels[kernel_id].ask()
            self._ask_queue.append(
                COMOCatCMAwM.Solution(
                    x=sol.x,
                    z=sol.z,
                    c=sol.c,
                    _v_raw=sol._v_raw,
                    _kernel_id=kernel_id,
                    _incumbent_id=None,
                )
            )

    def _enqueue_kernel_incumbent(self, kernel_id: int) -> None:
        """Enqueue the incumbent solution of a specific kernel for evaluation."""
        incumbent_sol = self.calc_incumbent_solution(kernel_id)
        self._ask_queue.append(
            COMOCatCMAwM.Solution(
                x=incumbent_sol.x,
                z=incumbent_sol.z,
                c=incumbent_sol.c,
                _v_raw=incumbent_sol._v_raw,
                _kernel_id=None,
                _incumbent_id=kernel_id,
            )
        )

    def _update_validation(self) -> bool:
        kid = self.current_kernel_id
        # all other kernels' incumbents must be evaluated
        if (self._incumbents_mask | (1 << kid)) != (1 << self._kernel_size) - 1:
            return False
        # must have exactly one full batch of tells for this kernel
        if len(self._tell_lists[kid]) < self._kernels[kid].population_size:
            return False
        elif len(self._tell_lists[kid]) > self._kernels[kid].population_size:
            raise RuntimeError(
                f"tell-list overflow for kernel {kid}: "
                f"size={len(self._tell_lists[kid])} > "
                f"population_size={self._kernels[kid].population_size}."
            )
        return True

    def _kernel_tell(self, kernel_id: int) -> None:
        assert (
            len(self._tell_lists[kernel_id]) == self._kernels[kernel_id].population_size
        ), "Tell list size does not match the corresponding kernel's population size."
        expected_mask = (1 << self._kernel_size) - 1
        assert (
            self._incumbents_mask | (1 << kernel_id)
        ) == expected_mask, (
            "All incumbents other than kernel_id must already be evaluated."
        )

        solutions = []
        for sol in self._tell_lists[kernel_id]:
            catcmawm_sol = CatCMAwM.Solution(
                sol[0].x, sol[0].z, sol[0].c, sol[0]._v_raw
            )
            obj = self._coerce_objective_space_point(sol[1])
            uhvi = self._UHVI(obj, kernel_id)
            # UHVI is maximized; negate for CatCMAwM's minimization
            solutions.append((catcmawm_sol, -uhvi))

        self._kernels[kernel_id].tell(solutions)
        self._tell_lists[kernel_id].clear()

    def ask(self) -> COMOCatCMAwM.Solution:
        """Retrieve a candidate solution from the internal evaluation queue.

        Returns:
            Solution: A Solution object containing continuous (x),
            integer (z), and/or categorical (c) variables.
        """
        if not self._ask_queue:
            raise AskQueueEmpty(
                "No solutions available to ask. Check ask_queue_size "
                "or iterate with ask_iter()."
            )

        sol = self._ask_queue.popleft()
        return sol

    def ask_iter(self, limit: int | None = None) -> Iterator[COMOCatCMAwM.Solution]:
        """Yield solutions until the ask-queue is empty or 'limit' is reached.
        Does not raise even when the queue becomes empty."""
        i = 0
        while self.ask_queue_size and (limit is None or i < limit):
            yield self.ask()
            i += 1

    @overload
    def tell(  # NOQA: E704
        self, observation: Tuple[COMOCatCMAwM.Solution, Sequence[float]]
    ) -> None: ...

    @overload
    def tell(  # NOQA: E704
        self, observation: List[Tuple[COMOCatCMAwM.Solution, Sequence[float]]]
    ) -> None: ...

    def tell(
        self,
        observation: (
            Tuple[COMOCatCMAwM.Solution, Sequence[float]]
            | List[Tuple[COMOCatCMAwM.Solution, Sequence[float]]]
        ),
    ) -> None:
        """Update the optimizer with the evaluation results of candidate solutions.

        Args:
            observation: A single (Solution, objective_values) pair or
            a list of such pairs.
        """
        # normalize to a list of pairs
        if isinstance(observation, list):
            solutions = observation
        elif isinstance(observation, tuple) and len(observation) == 2:
            solutions = [observation]
        else:
            raise TypeError(
                "tell expects either a (Solution, Sequence[float]) or "
                "a list of such tuples."
            )

        for sol in solutions:
            obj = self._coerce_objective_space_point(sol[1])
            self._update_ideal_nadir(obj)
            kernel_id = sol[0]._kernel_id
            incumbent_id = sol[0]._incumbent_id
            if kernel_id is not None and incumbent_id is None:
                self._tell_lists[kernel_id].append(sol)
            elif kernel_id is None and incumbent_id is not None:
                self._incumbents_sol[incumbent_id] = CatCMAwM.Solution(
                    sol[0].x, sol[0].z, sol[0].c, sol[0]._v_raw
                )
                if self._incumbents_obj is None:
                    raise RuntimeError(
                        "Incumbent objectives buffer is not initialized."
                    )
                self._incumbents_obj[incumbent_id] = sol[1]
                # Set evaluated flag for this incumbent
                self._incumbents_mask |= 1 << incumbent_id
            else:
                pass

        if self._update_validation():
            self._kernel_tell(self.current_kernel_id)
            self._enqueue_kernel_incumbent(self.current_kernel_id)
            # Clear evaluated flag for the current kernel
            self._incumbents_mask &= ~(1 << self.current_kernel_id)
            self._increment_kernel_index()
            self._enqueue_kernel_asks(self.current_kernel_id)
