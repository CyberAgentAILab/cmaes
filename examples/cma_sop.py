import numpy as np
from cmaes.cmasop import CMASoP


def example1():
    """
    example with benchmark on sets of points
    """

    # number of total dimensions
    dim = 10

    # number of dimensions in each subspace
    subspace_dim = 2

    # number of points in each subspace
    point_num = 10

    # objective function
    def quadratic(x):
        coef = 1000 ** (np.arange(dim) / float(dim - 1))
        return np.sum((x * coef) ** 2)

    # sets_of_points (on [-5, 5])
    discrete_subspace_num = dim // subspace_dim
    sets_of_points = (2 * np.random.rand(discrete_subspace_num, point_num, subspace_dim) - 1) * 5

    # add the optimal solution (for benchmark function)
    sets_of_points[:, -1] = np.zeros(subspace_dim)
    np.random.shuffle(sets_of_points)

    # optimizer (CMA-ES-SoP)
    optimizer = CMASoP(
        sets_of_points=sets_of_points,
        mean=np.random.rand(dim) * 4 + 1,
        sigma=2.0,
    )

    best_eval = np.inf
    eval_count = 0

    for generation in range(200):
        solutions = []
        for _ in range(optimizer.population_size):
            # Ask a parameter
            x, enc_x = optimizer.ask()
            value = quadratic(enc_x)

            # save best eval
            best_eval = np.min((best_eval, value))
            eval_count += 1

            solutions.append((x, value))

        # Tell evaluation values.
        optimizer.tell(solutions)

        print(f"#{generation} ({best_eval} {eval_count})")

        if best_eval < 1e-4 or optimizer.should_stop():
            break


def example2():
    """
    example with benchmark on mixed variable (sets of points and continuous variable)
    """

    # number of total dimensions
    dim = 10

    # number of dimensions in each subspace
    subspace_dim = 2

    # number of points in each subspace
    point_num = 10

    # objective function
    def quadratic(x):
        coef = 1000 ** (np.arange(dim) / float(dim - 1))
        return np.sum((x * coef) ** 2)

    # sets_of_points (on [-5, 5])
    # almost half of the subspaces are continuous spaces
    discrete_subspace_num = (dim // 2) // subspace_dim
    sets_of_points = (2 * np.random.rand(discrete_subspace_num, point_num, subspace_dim) - 1) * 5

    # add the optimal solution (for benchmark function)
    sets_of_points[:, -1] = np.zeros(subspace_dim)
    np.random.shuffle(sets_of_points)

    # optimizer (CMA-ES-SoP)
    optimizer = CMASoP(
        sets_of_points=sets_of_points,
        mean=np.random.rand(dim) * 4 + 1,
        sigma=2.0,
    )

    best_eval = np.inf
    eval_count = 0

    for generation in range(200):
        solutions = []
        for _ in range(optimizer.population_size):
            # Ask a parameter
            x, enc_x = optimizer.ask()
            value = quadratic(enc_x)

            # save best eval
            best_eval = np.min((best_eval, value))
            eval_count += 1

            solutions.append((x, value))

        # Tell evaluation values.
        optimizer.tell(solutions)

        print(f"#{generation} ({best_eval} {eval_count})")

        if best_eval < 1e-4 or optimizer.should_stop():
            break


def example3():
    """
    example with benchmark on mixed variable
    (continuous variable and sets of points with different numbers of dimensions and points)
    """

    # numbers of dimensions in each subspace
    subspace_dim_list = [2, 3, 5]
    cont_dim = 10

    # numbers of points in each subspace
    point_num_list = [10, 20, 40]

    # number of total dimensions
    dim = int(np.sum(subspace_dim_list) + cont_dim)

    # objective function
    def quadratic(x):
        coef = 1000 ** (np.arange(dim) / float(dim - 1))
        return np.sum((coef * x) ** 2)

    # sets_of_points (on [-5, 5])
    discrete_subspace_num = len(subspace_dim_list)
    sets_of_points = [
        (2 * np.random.rand(point_num_list[i], subspace_dim_list[i]) - 1) * 5
        for i in range(discrete_subspace_num)
    ]

    # add the optimal solution (for benchmark function)
    for i in range(discrete_subspace_num):
        sets_of_points[i][-1] = np.zeros(subspace_dim_list[i])
        np.random.shuffle(sets_of_points[i])

    # optimizer (CMA-ES-SoP)
    optimizer = CMASoP(
        sets_of_points=sets_of_points,
        mean=np.random.rand(dim) * 4 + 1,
        sigma=2.0,
    )

    best_eval = np.inf
    eval_count = 0

    for generation in range(400):
        solutions = []
        for _ in range(optimizer.population_size):
            # Ask a parameter
            x, enc_x = optimizer.ask()
            value = quadratic(enc_x)

            # save best eval
            best_eval = np.min((best_eval, value))
            eval_count += 1

            solutions.append((x, value))

        # Tell evaluation values.
        optimizer.tell(solutions)

        print(f"#{generation} ({best_eval} {eval_count})")

        if best_eval < 1e-4 or optimizer.should_stop():
            break


if __name__ == "__main__":
    example1()
    example2()
    # example3()
