# examples/ipop_sep_cma.py
import argparse
import numpy as np
from cmaes import IPOPSepCMA
from benchmark.functions import FUNCS  # sphere, rastrigin, rosenbrock, himmelblau, six_hump_camel, ellipsoid, ackley

def make_bounds(dim: int, low: float = -5.0, high: float = 5.0) -> np.ndarray:
    b = np.empty((dim, 2), dtype=float)
    b[:, 0], b[:, 1] = low, high
    return b

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--func", choices=sorted(FUNCS.keys()), default="sphere")
    p.add_argument("--dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stage-max-generations", type=int, default=None)
    p.add_argument("--budget", type=int, default=20000)         # NEW
    p.add_argument("--print-every", type=int, default=200)      # NEW (evals)
    args = p.parse_args()

    f = FUNCS[args.func]
    dim = 2 if args.func in ("himmelblau", "six_hump_camel") else args.dim
    bounds = make_bounds(dim, -5.0, 5.0)

    opt = IPOPSepCMA(
        mean=np.ones(dim) * 3.0,
        sigma=2.0,
        bounds=bounds,
        seed=args.seed,
        patience=args.patience,
        tol_rel_improve=1e-8,
        min_sigma=1e-8,
        stage_max_generations=args.stage_max_generations,
        max_restarts=10,
        # if you added cooldown in the class:
        # min_stage_generations=8,
    )

    evals = 0
    print(f"Starting IPOP-SepCMA on {args.func} (dim={dim}) with budget={args.budget}")
    while True:
        batch = []
        for _ in range(opt.population_size):
            x = opt.ask()
            fx = f(x)
            batch.append((x, fx))
            evals += 1
            if evals >= args.budget:
                break

        opt.tell(batch)

        # progress print
        if evals % args.print_every == 0 or evals >= args.budget:
            print(f"[{args.func}] evals={evals:6d} best={opt.best_f: .3e} "
                  f"gen={opt.generation:4d} restarts={opt.restart_count}")

        # robust exit conditions
        if evals >= args.budget:
            break
        # outer wrapper stop (e.g., max restarts)
        if getattr(opt, "should_stop", lambda: False)():
            break
        # inner optimizer stop (if you want to honor it too)
        if hasattr(opt, "_opt") and getattr(opt._opt, "should_stop", lambda: False)():
            break

    print(f"Done. Final best={opt.best_f:.6e}, evals={evals}, restarts={opt.restart_count}")

if __name__ == "__main__":
    main()
