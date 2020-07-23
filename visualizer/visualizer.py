"""
Usage:

  python3 visualizer/visualizer.py --function six-hump-camel
"""
import argparse
import numpy as np
from scipy import stats

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import rcParams

from cmaes._cma import CMA

parser = argparse.ArgumentParser()
parser.add_argument(
    "--function",
    choices=["quadratic", "himmelblau", "rosenbrock", "six-hump-camel"],
    default="rosenbrock",
)
args = parser.parse_args()

optimizer = CMA(
    mean=np.zeros(2), sigma=8 / 6, bounds=np.array([[-4, 4], [-4, 4]]), seed=1,
)
solutions = []

rcParams["figure.figsize"] = 10, 5
fig, (ax1, ax2) = plt.subplots(1, 2)

color_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "yellow": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
}
bw = LinearSegmentedColormap("BlueWhile", color_dict)


def himmelbleu(x1, x2):
    return (x1 ** 2 + x2 - 11.0) ** 2 + (x1 + x2 ** 2 - 7.0) ** 2


def himmelbleu_contour(x1, x2):
    return np.log(himmelbleu(x1, x2) + 1)


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def quadratic_contour(x1, x2):
    return np.log(quadratic(x1, x2) + 1)


def rosenbrock(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


def rosenbrock_contour(x1, x2):
    return np.log(rosenbrock(x1, x2) + 1)


def six_hump_camel(x1, x2):
    return (
        (4 - 2.1 * (x1 ** 2) + (x1 ** 4) / 3) * (x1 ** 2)
        + x1 * x2
        + (-4 + 4 * x2 ** 2) * (x2 ** 2)
    )


def six_hump_camel_contour(x1, x2):
    return np.log(six_hump_camel(x1, x2) + 1.0316)


if args.function == "quadratic":
    fig.suptitle("CMA Evolution Strategy for quadratic function")
    objective = quadratic
    contour_function = quadratic_contour
    global_minimums = [
        (3.0, -2.0),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -4, 4
    x2_lower_bound, x2_upper_bound = -4, 4
elif args.function == "himmelblau":
    fig.suptitle("CMA Evolution Strategy for Himmelblau function")
    objective = himmelbleu
    contour_function = himmelbleu_contour
    global_minimums = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -4, 4
    x2_lower_bound, x2_upper_bound = -4, 4
elif args.function == "rosenbrock":
    # https://www.sfu.ca/~ssurjano/rosen.html
    fig.suptitle("CMA Evolution Strategy for Rosenbrock function")
    objective = rosenbrock
    contour_function = rosenbrock_contour
    global_minimums = [
        (1, 1),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -5, 10
    x2_lower_bound, x2_upper_bound = -5, 10
elif args.function == "six-hump-camel":
    # https://www.sfu.ca/~ssurjano/camel6.html
    fig.suptitle("CMA Evolution Strategy for Six-hump camel function")
    objective = six_hump_camel
    contour_function = six_hump_camel_contour
    global_minimums = [
        (0.0898, -0.7126),
        (-0.0898, 0.7126),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -3, 3
    x2_lower_bound, x2_upper_bound = -2, 2
else:
    raise ValueError("invalid function type")


def init():
    ax1.set_xlim(x1_lower_bound, x1_upper_bound)
    ax1.set_ylim(x2_lower_bound, x2_upper_bound)
    ax2.set_xlim(x1_lower_bound, x1_upper_bound)
    ax2.set_ylim(x2_lower_bound, x2_upper_bound)

    # Plot 4 local minimum value
    for m in global_minimums:
        ax1.plot(m[0], m[1], "y*", ms=10)
        ax2.plot(m[0], m[1], "y*", ms=10)

    # Plot contour of himmelbleu function
    x1 = np.arange(x1_lower_bound, x1_upper_bound, 0.01)
    x2 = np.arange(x2_lower_bound, x2_upper_bound, 0.01)
    x1, x2 = np.meshgrid(x1, x2)

    ax1.contour(x1, x2, contour_function(x1, x2), 30, cmap=bw)


def update(frame):
    global solutions
    if len(solutions) == optimizer.population_size:
        optimizer.tell(solutions)
        solutions = []

    x = optimizer.ask()
    evaluation = objective(x[0], x[1])

    solution = (
        x,
        evaluation,
    )
    solutions.append(solution)

    # Plot sample points
    ax1.plot(x[0], x[1], "o", c="r", label="2d", alpha=0.5)

    # Plot multivariate gaussian distribution of CMA-ES
    mu = optimizer._mean
    sigma = optimizer._C
    x, y = np.mgrid[
        x1_lower_bound:x1_upper_bound:0.01, x2_lower_bound:x2_upper_bound:0.01
    ]
    rv = stats.multivariate_normal(mu, sigma)
    pos = np.dstack((x, y))
    ax2.contourf(x, y, rv.pdf(pos))


def main():
    ani = animation.FuncAnimation(
        fig, update, frames=150, init_func=init, blit=False, interval=50
    )
    ani.save(f"./tmp/{args.function}.mp4")


if __name__ == "__main__":
    main()
