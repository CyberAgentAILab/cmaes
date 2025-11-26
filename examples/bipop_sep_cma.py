# examples/bipop_sep_cma.py
import argparse
import numpy as np
from cmaes import BIPOPSepCMA
from benchmark.functions import FUNCS

def make_bounds(dim: int, low: float = -5.0, high: float = 5.0) -> np.ndarray:
    b = np.empty((dim, 2), dtype=float)
    b[:, 0], b[:, 1] = low, high
    return b

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--func", choices=sorted(FUNCS.keys()), default="rastrigin")
    p.add_argument("--dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--stage-max-generations", type=int, default=150)
    p.add_argument("--budget", type=int, default=20000)         # NEW
    p.add_argument("--print-every", type=int, default=200)      # NEW
    args = p.parse_args()

    f = FUNCS[args.func]
    dim = 2 if args.func in ("himmelblau", "six_hump_camel") else args.dim
    bounds = make_bounds(dim, -5.0, 5.0)

    opt = BIPOPSepCMA(
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
    print(f"Starting BIPOP-SepCMA on {args.func} (dim={dim}) with budget={args.budget}")
    while True:
        batch = []
        for _ in range(opt.population_size):
            x = opt.ask()
            fx = f(x)
            batch.append((x, fx))
            evals += 1
            if evals >= args.budget:
                break

        # ✅ NEW safe guard
        if len(batch) < opt.population_size:
            break

        opt.tell(batch)

        if evals % args.print_every == 0 or evals >= args.budget:
            regime = getattr(opt, "regime", "?")
            print(f"[{args.func}] evals={evals:6d} best={opt.best_f: .3e} "
                  f"gen={opt.generation:4d} regime={regime} restarts={opt.restart_count}")

        if evals >= args.budget:
            break
        if getattr(opt, "should_stop", lambda: False)():
            break
        if hasattr(opt, "_opt") and getattr(opt._opt, "should_stop", lambda: False)():
            break

    print(f"Done. Final best={opt.best_f:.6e}, evals={evals}, restarts={opt.restart_count}")

if __name__ == "__main__":
    main()
