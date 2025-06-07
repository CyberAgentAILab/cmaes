import numpy as np
from cmaes import CatCMAwM


def SphereIntCOM(x, z, c):
    return sum(x * x) + sum(z * z) + len(c) - sum(c[:, 0])


def SphereInt(x, z):
    return sum(x * x) + sum(z * z)


def SphereCOM(x, c):
    return sum(x * x) + len(c) - sum(c[:, 0])


def f_cont_int_cat():
    # [lower_bound, upper_bound] for each continuous variable
    X = [[-5, 5], [-5, 5]]
    # possible values for each integer variable
    Z = [[-1, 0, 1], [-2, -1, 0, 1, 2]]
    # number of categories for each categorical variable
    C = [3, 3]

    optimizer = CatCMAwM(x_space=X, z_space=Z, c_space=C)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            sol = optimizer.ask()
            value = SphereIntCOM(sol.x, sol.z, sol.c)
            solutions.append((sol, value))
            print(f"#{generation} {sol} evaluation: {value}")
        optimizer.tell(solutions)


def f_cont_int():
    # [lower_bound, upper_bound] for each continuous variable
    X = [[-np.inf, np.inf], [-np.inf, np.inf]]
    # possible values for each integer variable
    Z = [[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]]

    # initial distribution parameters (Optional)
    init_mean = np.ones(len(X) + len(Z))
    init_cov = np.diag(np.ones(len(X) + len(Z)))
    init_sigma = 1.0

    optimizer = CatCMAwM(
        x_space=X, z_space=Z, mean=init_mean, cov=init_cov, sigma=init_sigma
    )

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            sol = optimizer.ask()
            value = SphereInt(sol.x, sol.z)
            solutions.append((sol, value))
            print(f"#{generation} {sol} evaluation: {value}")
        optimizer.tell(solutions)


def f_cont_cat():
    # [lower_bound, upper_bound] for each continuous variable
    X = [[-5, 5], [-5, 5]]
    # number of categories for each categorical variable
    C = [3, 5]

    # initial distribution parameters (Optional)
    init_cat_param = np.array(
        [
            [0.5, 0.3, 0.2, 0.0, 0.0],  # zero-padded at the end
            [0.2, 0.2, 0.2, 0.2, 0.2],  # each row must sum to 1
        ]
    )

    optimizer = CatCMAwM(x_space=X, c_space=C, cat_param=init_cat_param)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            sol = optimizer.ask()
            value = SphereCOM(sol.x, sol.c)
            solutions.append((sol, value))
            print(f"#{generation} {sol} evaluation: {value}")
        optimizer.tell(solutions)


if __name__ == "__main__":
    f_cont_int_cat()
    # f_cont_int()
    # f_cont_cat()
