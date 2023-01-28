import torch
import numpy as np

from cmaes import CMA


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def main():
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)
    torch_sobol = torch.quasirandom.SobolEngine(2, scramble=True, seed=None)
    device = torch.device("cpu")
    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")

    while True:
        solutions = []
        for _ in range(optimizer.population_size):

            # Draw uniform quasi-random variables and transform to standard normal distribution
            # by Box-Muller method.
            u1 = torch_sobol.draw(1).to(dtype=torch.float64, device=device).cpu().detach().numpy()[0]
            u2 = torch_sobol.draw(1).to(dtype=torch.float64, device=device).cpu().detach().numpy()[0]
            sobol_z = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

            x = optimizer.ask(z=sobol_z)
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(
                f"{optimizer.generation:3d}  {value:10.5f}"
                f"  {x[0]:6.2f}  {x[1]:6.2f}"
            )
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
