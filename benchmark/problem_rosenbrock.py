rosenbrock_problem.pyfrom kurobako import problem
from kurobako.problem import Problem

from typing import List
from typing import Optional


class RosenbrockEvaluator(problem.Evaluator):
    """
    See https://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(self, params: List[Optional[float]]):
        self._x1, self._x2 = params
        self._current_step = 0

    def evaluate(self, next_step: int) -> List[float]:
        self._current_step = 1
        value = 100 * (self._x2 - self._x1 ** 2) ** 2 + (self._x1 - 1) ** 2
        return [value]

    def current_step(self) -> int:
        return self._current_step


class RosenbrockProblem(problem.Problem):
    def create_evaluator(
        self, params: List[Optional[float]]
    ) -> Optional[problem.Evaluator]:
        return RosenbrockEvaluator(params)


class RosenbrockProblemFactory(problem.ProblemFactory):
    def create_problem(self, seed: int) -> Problem:
        return RosenbrockProblem()

    def specification(self) -> problem.ProblemSpec:
        params = [
            problem.Var("x1", problem.ContinuousRange(-5, 10)),
            problem.Var("x2", problem.ContinuousRange(-5, 10)),
        ]
        return problem.ProblemSpec(
            name="Rosenbrock Function",
            params=params,
            values=[problem.Var("Rosenbrock")],
        )


if __name__ == "__main__":
    runner = problem.ProblemRunner(RosenbrockProblemFactory())
    runner.run()
