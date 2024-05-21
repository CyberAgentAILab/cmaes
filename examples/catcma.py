import numpy as np
from cmaes import CatCMA


def sphere_com(x, c):
    dim_co = len(x)
    dim_ca = len(c)
    if dim_co < 2:
        raise ValueError("dimension must be greater one")
    sphere = sum(x * x)
    com = dim_ca - sum(c[:, 0])
    return sphere + com


def rosenbrock_clo(x, c):
    dim_co = len(x)
    dim_ca = len(c)
    if dim_co < 2:
        raise ValueError("dimension must be greater one")
    rosenbrock = sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)
    clo = dim_ca - (c[:, 0].argmin() + c[:, 0].prod() * dim_ca)
    return rosenbrock + clo


def mc_proximity(x, c, cat_num):
    dim_co = len(x)
    dim_ca = len(c)
    if dim_co < 2:
        raise ValueError("dimension must be greater one")
    if dim_co != dim_ca:
        raise ValueError(
            "number of dimensions of continuous and categorical variables "
            "must be equal in mc_proximity"
        )

    c_index = np.argmax(c, axis=1) / cat_num
    return sum((x - c_index) ** 2) + sum(c_index)


if __name__ == "__main__":
    cont_dim = 5
    cat_dim = 5
    cat_num = np.array([3, 4, 5, 5, 5])
    # cat_num = 3 * np.ones(cat_dim, dtype=np.int64)
    optimizer = CatCMA(mean=3.0 * np.ones(cont_dim), sigma=1.0, cat_num=cat_num)

    for generation in range(200):
        solutions = []
        for _ in range(optimizer.population_size):
            x, c = optimizer.ask()
            value = mc_proximity(x, c, cat_num)
            if generation % 10 == 0:
                print(f"#{generation} {value}")
            solutions.append(((x, c), value))
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break
