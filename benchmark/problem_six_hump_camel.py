from kurobako import problem
from kurobako.problem import Problem

from typing import List
from typing import Optional


class SixHumpCamelEvaluator(problem.Evaluator):
    """
    See https://www.sfu.ca/~ssurjano/camel6.html
    """

    def __init__(self, params: List[Optional[float]]):
        self._x1, self._x2 = params
        self._current_step = 0

    def evaluate(self, next_step: int) -> List[float]:
        self._current_step = 1
        value = (
            (4 - 2.1 * (self._x1**2) + (self._x1**4) / 3) * (self._x1**2)
            + self._x1 * self._x2
            + (-4 + 4 * self._x2**2) * (self._x2**2)
        )
        return [value]

    def current_step(self) -> int:
        return self._current_step


class SixHumpCamelProblem(problem.Problem):
    def create_evaluator(
        self, params: List[Optional[float]]
    ) -> Optional[problem.Evaluator]:
        return SixHumpCamelEvaluator(params)


class SixHumpCamelProblemFactory(problem.ProblemFactory):
    def create_problem(self, seed: int) -> Problem:
        return SixHumpCamelProblem()

    def specification(self) -> problem.ProblemSpec:
        params = [
            problem.Var("x1", problem.ContinuousRange(-5, 10)),
            problem.Var("x2", problem.ContinuousRange(-5, 10)),
        ]
        return problem.ProblemSpec(
            name="Six-Hump Camel Function",
            params=params,
            values=[problem.Var("Six-Hump Camel")],
        )


if __name__ == "__main__":
    runner = problem.ProblemRunner(SixHumpCamelProblemFactory())
    runner.run()
