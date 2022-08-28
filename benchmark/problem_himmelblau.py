from kurobako import problem
from kurobako.problem import Problem

from typing import List
from typing import Optional


class HimmelblauEvaluator(problem.Evaluator):
    def __init__(self, params: List[Optional[float]]):
        self._x1, self._x2 = params
        self._current_step = 0

    def evaluate(self, next_step: int) -> List[float]:
        self._current_step = 1
        value = (self._x1**2 + self._x2 - 11.0) ** 2 + (
            self._x1 + self._x2**2 - 7.0
        ) ** 2
        return [value]

    def current_step(self) -> int:
        return self._current_step


class HimmelblauProblem(problem.Problem):
    def create_evaluator(
        self, params: List[Optional[float]]
    ) -> Optional[problem.Evaluator]:
        return HimmelblauEvaluator(params)


class HimmelblauProblemFactory(problem.ProblemFactory):
    def create_problem(self, seed: int) -> Problem:
        return HimmelblauProblem()

    def specification(self) -> problem.ProblemSpec:
        params = [
            problem.Var("x1", problem.ContinuousRange(-4, 4)),
            problem.Var("x2", problem.ContinuousRange(-4, 4)),
        ]
        return problem.ProblemSpec(
            name="Himmelblau Function",
            params=params,
            values=[problem.Var("Himmelblau")],
        )


if __name__ == "__main__":
    runner = problem.ProblemRunner(HimmelblauProblemFactory())
    runner.run()
