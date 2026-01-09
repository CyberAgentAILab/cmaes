import numpy as np
from cmaes import COMOCatCMAwM


def DSIntLFTL(x, z, c, cat_num):
    Sphere1 = sum((x / 10) ** 2) / len(x)
    Sphere2 = sum((x / 10 - 1) ** 2) / len(x)
    SphereInt1 = sum((z / 10) ** 2) / len(z)
    SphereInt2 = sum((z / 10 - 1) ** 2) / len(z)
    c_idx = c.argmax(axis=1)
    LF = (len(c) - (c_idx == 0).cumprod().sum()) / len(c)
    TL = (len(c) - (c_idx == np.asarray(cat_num) - 1)[::-1].cumprod().sum()) / len(c)
    obj1 = Sphere1 + SphereInt1 + LF
    obj2 = Sphere2 + SphereInt2 + TL
    return [obj1, obj2]


if __name__ == "__main__":
    # [lower_bound, upper_bound] for each continuous variable
    X = [[-5, 15]] * 3
    # possible values for each integer variable
    Z = [range(-5, 16)] * 3
    # number of categories for each categorical variable
    C = [5] * 3

    optimizer = COMOCatCMAwM(x_space=X, z_space=Z, c_space=C)

    evals = 0
    while evals < 7000:
        solutions = []
        for sol in optimizer.ask_iter():
            value = DSIntLFTL(sol.x, sol.z, sol.c, C)
            evals += 1
            solutions.append((sol, value))
        optimizer.tell(solutions)
        print(evals, optimizer.incumbent_objectives)
