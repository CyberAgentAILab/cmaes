import sys
import numpy as np

from kurobako import problem
from kurobako.problem import Problem

from typing import List
from typing import Optional


class RastriginEvaluator(problem.Evaluator):
    def __init__(self, params: List[Optional[float]]):
        self.n = len(params)
        self.x = np.array(params, dtype=float)
        self._current_step = 0

    def evaluate(self, next_step: int) -> List[float]:
        self._current_step = 1
        value = 10 * self.n + np.sum(self.x**2 - 10 * np.cos(2 * np.pi * self.x))
        return [value]

    def current_step(self) -> int:
        return self._current_step


class RastriginProblem(problem.Problem):
    def create_evaluator(
        self, params: List[Optional[float]]
    ) -> Optional[problem.Evaluator]:
        return RastriginEvaluator(params)


class RastriginProblemFactory(problem.ProblemFactory):
    def __init__(self, dim):
        self.dim = dim

    def create_problem(self, seed: int) -> Problem:
        return RastriginProblem()

    def specification(self) -> problem.ProblemSpec:
        params = [
            problem.Var(f"x{i + 1}", problem.ContinuousRange(-5.12, 5.12))
            for i in range(self.dim)
        ]
        return problem.ProblemSpec(
            name=f"Rastrigin (dim={self.dim})",
            params=params,
            values=[problem.Var("Rastrigin")],
        )


if __name__ == "__main__":
    dim = int(sys.argv[1]) if len(sys.argv) == 2 else 2
    runner = problem.ProblemRunner(RastriginProblemFactory(dim))
    runner.run()
