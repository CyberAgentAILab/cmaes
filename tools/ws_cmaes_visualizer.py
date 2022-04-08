"""
Usage:
  python3 tools/ws_cmaes_visualizer.py OPTIONS

Optional arguments:
  -h, --help            show this help message and exit
  --function {quadratic,himmelblau,rosenbrock,six-hump-camel,sphere,rot-ellipsoid}
  --seed SEED
  --alpha ALPHA
  --gamma GAMMA
  --frames FRAMES
  --interval INTERVAL
  --pop-per-frame POP_PER_FRAME

Example:
  python3 ws_cmaes_visualizer.py --function rot-ellipsoid
"""
import argparse
import math

import numpy as np
from scipy import stats

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import rcParams

from cmaes import get_warm_start_mgd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--function",
    choices=[
        "quadratic",
        "himmelblau",
        "rosenbrock",
        "six-hump-camel",
        "sphere",
        "rot-ellipsoid",
    ],
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--frames",
    type=int,
    default=100,
)
parser.add_argument(
    "--interval",
    type=int,
    default=20,
)
parser.add_argument(
    "--pop-per-frame",
    type=int,
    default=10,
)
args = parser.parse_args()

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
    return (x1**2 + x2 - 11.0) ** 2 + (x1 + x2**2 - 7.0) ** 2


def himmelbleu_contour(x1, x2):
    return np.log(himmelbleu(x1, x2) + 1)


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def quadratic_contour(x1, x2):
    return np.log(quadratic(x1, x2) + 1)


def rosenbrock(x1, x2):
    return 100 * (x2 - x1**2) ** 2 + (x1 - 1) ** 2


def rosenbrock_contour(x1, x2):
    return np.log(rosenbrock(x1, x2) + 1)


def six_hump_camel(x1, x2):
    return (
        (4 - 2.1 * (x1**2) + (x1**4) / 3) * (x1**2)
        + x1 * x2
        + (-4 + 4 * x2**2) * (x2**2)
    )


def six_hump_camel_contour(x1, x2):
    return np.log(six_hump_camel(x1, x2) + 1.0316)


def sphere(x1, x2):
    offset = 0.6
    return (x1 - offset) ** 2 + (x2 - offset) ** 2


def sphere_contour(x1, x2):
    return np.log(sphere(x1, x2) + 1)


def ellipsoid(x1, x2):
    offset = 0.6
    scale = 5**2
    return (x1 - offset) ** 2 + scale * (x2 - offset) ** 2


def rot_ellipsoid(x1, x2):
    rot_x1 = math.sqrt(3.0) / 2.0 * x1 + 1.0 / 2.0 * x2
    rot_x2 = 1.0 / 2.0 * x1 + math.sqrt(3.0) / 2.0 * x2
    return ellipsoid(rot_x1, rot_x2)


def rot_ellipsoid_contour(x1, x2):
    return np.log(rot_ellipsoid(x1, x2) + 1)


function_name = ""
if args.function == "quadratic":
    function_name = "Quadratic function"
    objective = quadratic
    contour_function = quadratic_contour
    global_minimums = [
        (3.0, -2.0),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -4, 4
    x2_lower_bound, x2_upper_bound = -4, 4
elif args.function == "himmelblau":
    function_name = "Himmelblau function"
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
    function_name = "Rosenbrock function"
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
    function_name = "Six-hump camel function"
    objective = six_hump_camel
    contour_function = six_hump_camel_contour
    global_minimums = [
        (0.0898, -0.7126),
        (-0.0898, 0.7126),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -3, 3
    x2_lower_bound, x2_upper_bound = -2, 2
elif args.function == "sphere":
    function_name = "Sphere function with offset=0.6"
    objective = sphere
    contour_function = sphere_contour
    global_minimums = [
        (0.6, 0.6),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = 0, 1
    x2_lower_bound, x2_upper_bound = 0, 1
elif args.function == "rot-ellipsoid":
    function_name = "Rot Ellipsoid function with offset=0.6"
    objective = rot_ellipsoid
    contour_function = rot_ellipsoid_contour

    global_minimums = []
    # input domain
    x1_lower_bound, x1_upper_bound = 0, 1
    x2_lower_bound, x2_upper_bound = 0, 1
else:
    raise ValueError("invalid function type")


seed = args.seed
rng = np.random.RandomState(seed)
solutions = []


def init():
    ax1.set_xlim(x1_lower_bound, x1_upper_bound)
    ax1.set_ylim(x2_lower_bound, x2_upper_bound)
    ax2.set_xlim(x1_lower_bound, x1_upper_bound)
    ax2.set_ylim(x2_lower_bound, x2_upper_bound)

    # Plot 4 local minimum value
    for m in global_minimums:
        ax1.plot(m[0], m[1], "y*", ms=10)
        ax2.plot(m[0], m[1], "y*", ms=10)

    # Plot contour of the function
    x1 = np.arange(x1_lower_bound, x1_upper_bound, 0.01)
    x2 = np.arange(x2_lower_bound, x2_upper_bound, 0.01)
    x1, x2 = np.meshgrid(x1, x2)

    ax1.contour(x1, x2, contour_function(x1, x2), 30, cmap=bw)


def update(frame):
    global solutions

    for i in range(args.pop_per_frame):
        x1 = (x1_upper_bound - x1_lower_bound) * rng.random() + x1_lower_bound
        x2 = (x2_upper_bound - x2_lower_bound) * rng.random() + x2_lower_bound

        evaluation = objective(x1, x2)

        # Plot sample points
        ax1.plot(x1, x2, "o", c="r", label="2d", alpha=0.5)

        solution = (
            np.array([x1, x2], dtype=float),
            evaluation,
        )
        solutions.append(solution)

    # Update title
    fig.suptitle(
        f"WS-CMA-ES {function_name} with alpha={args.alpha} and gamma={args.gamma} (frame={frame})"
    )

    # Plot multivariate gaussian distribution of CMA-ES
    x, y = np.mgrid[
        x1_lower_bound:x1_upper_bound:0.01, x2_lower_bound:x2_upper_bound:0.01
    ]

    if math.floor(len(solutions) * args.alpha) > 1:
        mean, sigma, cov = get_warm_start_mgd(
            solutions, alpha=args.alpha, gamma=args.gamma
        )
        rv = stats.multivariate_normal(mean, cov)
        pos = np.dstack((x, y))
        ax2.contourf(x, y, rv.pdf(pos))

    if frame % 50 == 0:
        print(f"Processing frame {frame}")


def main():
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=args.frames,
        init_func=init,
        blit=False,
        interval=args.interval,
    )
    ani.save(f"./tmp/{args.function}.mp4")


if __name__ == "__main__":
    main()
