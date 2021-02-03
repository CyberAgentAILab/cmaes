"""
Usage:
  cmaes_visualizer.py OPTIONS

Optional arguments:
  -h, --help            show this help message and exit
  --function {quadratic,himmelblau,rosenbrock,six-hump-camel}
  --seed SEED
  --frames FRAMES
  --interval INTERVAL
  --pop-per-frame POP_PER_FRAME
  --restart-strategy {ipop,bipop}

Example:
  python3 cmaes_visualizer.py --function six-hump-camel --pop-per-frame 2

  python3 tools/cmaes_visualizer.py --function himmelblau \
    --restart-strategy ipop --frames 500 --interval 10 --pop-per-frame 6
"""
import argparse
import math

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
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
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
    default=1,
)
parser.add_argument(
    "--restart-strategy",
    choices=["ipop", "bipop"],
    default="",
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
else:
    raise ValueError("invalid function type")


seed = args.seed
bounds = np.array([[x1_lower_bound, x1_upper_bound], [x2_lower_bound, x2_upper_bound]])
sigma = (x1_upper_bound - x2_lower_bound) / 5
optimizer = CMA(mean=np.zeros(2), sigma=sigma, bounds=bounds, seed=seed)
solutions = []
trial_number = 0
rng = np.random.RandomState(seed)

# Variables for IPOP and BIPOP
inc_popsize = 2
n_restarts = 0  # A small restart doesn't count in the n_restarts
small_n_eval, large_n_eval = 0, 0
popsize0 = optimizer.population_size
poptype = "small"


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


def get_next_popsize():
    global optimizer, n_restarts, poptype, small_n_eval, large_n_eval
    if args.restart_strategy == "ipop":
        n_restarts += 1
        popsize = optimizer.population_size * inc_popsize
        print(f"Restart CMA-ES with popsize={popsize} at trial={trial_number}")
        return popsize
    elif args.restart_strategy == "bipop":
        n_eval = optimizer.population_size * optimizer.generation
        if poptype == "small":
            small_n_eval += n_eval
        else:  # poptype == "large"
            large_n_eval += n_eval

        if small_n_eval < large_n_eval:
            poptype = "small"
            popsize_multiplier = inc_popsize ** n_restarts
            popsize = math.floor(popsize0 * popsize_multiplier ** (rng.uniform() ** 2))
        else:
            poptype = "large"
            n_restarts += 1
            popsize = popsize0 * (inc_popsize ** n_restarts)
        print(
            f"Restart CMA-ES with popsize={popsize} ({poptype}) at trial={trial_number}"
        )
        return
    raise Exception("must not reach here")


def update(frame):
    global solutions, optimizer, trial_number
    if len(solutions) == optimizer.population_size:
        optimizer.tell(solutions)
        solutions = []

        if optimizer.should_stop():
            popsize = get_next_popsize()
            lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
            mean = lower_bounds + (rng.rand(2) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                seed=seed,
                population_size=popsize,
            )

    n_sample = min(optimizer.population_size - len(solutions), args.pop_per_frame)
    for i in range(n_sample):
        x = optimizer.ask()
        evaluation = objective(x[0], x[1])

        # Plot sample points
        ax1.plot(x[0], x[1], "o", c="r", label="2d", alpha=0.5)

        solution = (
            x,
            evaluation,
        )
        solutions.append(solution)
    trial_number += n_sample

    # Update title
    if args.restart_strategy == "ipop":
        fig.suptitle(
            f"IPOP-CMA-ES {function_name} trial={trial_number} "
            f"popsize={optimizer.population_size}"
        )
    elif args.restart_strategy == "bipop":
        fig.suptitle(
            f"BIPOP-CMA-ES {function_name} trial={trial_number} "
            f"popsize={optimizer.population_size} ({poptype})"
        )
    else:
        fig.suptitle(f"CMA-ES {function_name} trial={trial_number}")

    # Plot multivariate gaussian distribution of CMA-ES
    x, y = np.mgrid[
        x1_lower_bound:x1_upper_bound:0.01, x2_lower_bound:x2_upper_bound:0.01
    ]
    rv = stats.multivariate_normal(optimizer._mean, optimizer._C)
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
