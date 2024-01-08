from __future__ import annotations

import sys
import numpy as np

from kurobako import problem
from kurobako.problem import Problem

from typing import Optional


class SphereEvaluator(problem.Evaluator):
    def __init__(self, params: list[Optional[float]]):
        self.n = len(params)
        self.x = np.array(params, dtype=float)
        self._current_step = 0

    def evaluate(self, next_step: int) -> list[float]:
        self._current_step = 1
        value = np.mean(self.x**2)
        return [value]

    def current_step(self) -> int:
        return self._current_step


class SphereProblem(problem.Problem):
    def create_evaluator(
        self, params: list[Optional[float]]
    ) -> Optional[problem.Evaluator]:
        return SphereEvaluator(params)


class SphereProblemFactory(problem.ProblemFactory):
    def __init__(self, dim):
        self.dim = dim

    def create_problem(self, seed: int) -> Problem:
        return SphereProblem()

    def specification(self) -> problem.ProblemSpec:
        params = [
            problem.Var(f"x{i + 1}", problem.ContinuousRange(-5.12, 5.12))
            for i in range(self.dim)
        ]
        return problem.ProblemSpec(
            name=f"Sphere (dim={self.dim})",
            params=params,
            values=[problem.Var("Sphere")],
        )


if __name__ == "__main__":
    dim = int(sys.argv[1]) if len(sys.argv) == 2 else 2
    runner = problem.ProblemRunner(SphereProblemFactory(dim))
    runner.run()
