# benchmark/bench_hybrid.py
from __future__ import annotations
import argparse, csv, math, time, json
from pathlib import Path
from typing import Dict, Callable, List, Optional
import numpy as np

# ---------- Algorithms ----------
from cmaes import CMA, SepCMA, IPOPSepCMA
try:
    from cmaes import BIPOPSepCMA
    HAS_BIPOP_SEP = True
except Exception:
    HAS_BIPOP_SEP = False

# Full-CMA wrappers we add in Task 1:
try:
    from cmaes import IPOPFullCMA, BIPOPFullCMA
    HAS_FULL_WRAPPERS = True
except Exception:
    HAS_FULL_WRAPPERS = False

# ---------- Functions ----------
from benchmark.functions import FUNCS  # registry of callables

# Target thresholds for success/evals_to_target
TARGETS = {
    "sphere": 1e-10,
    "rastrigin": 200,
    "ackley": 1e-8,
    "rosenbrock": 200,
    "ellipsoid": 1e-8,
}

def normalize_dim_for_func(func: str, dim: int) -> int:
    if func in ("himmelblau", "six_hump_camel"):
        return 2
    return dim

def bounds_for_func(func: str, dim: int) -> Optional[np.ndarray]:
    # Common bounded domains
    if func in {"rastrigin", "ackley"}:
        b = np.empty((dim, 2), float); b[:,0], b[:,1] = -5.0, 5.0; return b
    if func in {"rosenbrock"}:
        b = np.empty((dim, 2), float); b[:,0], b[:,1] = -5.0, 10.0; return b
    if func in {"ellipsoid", "sphere"}:
        return None  # usually fine unbounded
    return None

def make_optimizer(algo: str, dim: int, seed: int, bounds: Optional[np.ndarray]):
    mean = np.ones(dim) * 3.0
    sigma = 2.0
    # Baselines
    if algo == "cma_full":
        return CMA(mean=mean, sigma=sigma, seed=seed, bounds=bounds)
    if algo == "sep":
        return SepCMA(mean=mean, sigma=sigma, seed=seed, bounds=bounds)
    # Hybrids (Sep)
    if algo == "ipop_sep":
        return IPOPSepCMA(mean=mean, sigma=sigma, seed=seed, bounds=bounds)
    if algo == "bipop_sep" and HAS_BIPOP_SEP:
        return BIPOPSepCMA(mean=mean, sigma=sigma, seed=seed, bounds=bounds)
    # Hybrids (Full-CMA)
    if algo == "ipop_full" and HAS_FULL_WRAPPERS:
        return IPOPFullCMA(mean=mean, sigma=sigma, seed=seed, bounds=bounds)
    if algo == "bipop_full" and HAS_FULL_WRAPPERS:
        return BIPOPFullCMA(mean=mean, sigma=sigma, seed=seed, bounds=bounds)
    raise ValueError(f"Unknown/unavailable algo: {algo}")

def run_once(func: str, dim: int, algo: str, seed: int, budget: int, print_every: int = 0) -> Dict[str, float]:
    f: Callable[[np.ndarray], float] = FUNCS[func]
    dim = normalize_dim_for_func(func, dim)
    bounds = bounds_for_func(func, dim)
    opt = make_optimizer(algo, dim, seed, bounds)

    target = TARGETS.get(func, None)
    evals = 0
    best = math.inf
    evals_to_target = math.inf
    t0 = time.time()

    while True:
        batch = []
        for _ in range(opt.population_size):
            x = opt.ask()
            fx = float(f(x))
            batch.append((x, fx))
            evals += 1
            if fx < best:
                best = fx
            if target is not None and fx <= target and math.isinf(evals_to_target):
                evals_to_target = evals
            if evals >= budget:
                break

        # never tell a partial batch
        if len(batch) < opt.population_size:
            break
        opt.tell(batch)

        if print_every and evals % print_every == 0:
            pass  # could print progress

        if evals >= budget: break
        if getattr(opt, "should_stop", lambda: False)(): break
        if hasattr(opt, "_opt") and getattr(opt._opt, "should_stop", lambda: False)(): break

    seconds = time.time() - t0
    restarts = getattr(opt, "restart_count", 0)
    success = 1 if (target is not None and best <= target) else 0

    params = {
        "seed": seed,
        "budget": budget,
        "bounds": "[-5,5]" if bounds is not None else "None",
        "target": target,
    }

    return dict(
        func=func, algo=algo, dim=dim, seed=seed,
        best=best, evals=evals, seconds=seconds, restarts=restarts,
        evals_to_target=evals_to_target, success=success, params=json.dumps(params),
    )

def parse_list(s: str) -> List[str]:  return [t.strip() for t in s.split(",") if t.strip()]
def parse_int_list(s: str) -> List[int]:  return [int(t.strip()) for t in s.split(",") if t.strip()]

def main():
    parser = argparse.ArgumentParser(description="CMA-ES benchmark runner")
    parser.add_argument("--funcs", type=parse_list,
        default="rosenbrock,rastrigin,ackley,sphere,ellipsoid")
    parser.add_argument("--dims", type=parse_int_list, default="10,20,50")
    parser.add_argument("--algos", type=parse_list,
        default="cma_full,ipop_full,bipop_full,sep,ipop_sep,bipop_sep")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="benchmark/logs")
    args = parser.parse_args()

    # validate availability
    avail = {"cma_full","sep","ipop_sep"} | ({"bipop_sep"} if HAS_BIPOP_SEP else set()) | ({"ipop_full","bipop_full"} if HAS_FULL_WRAPPERS else set())
    for a in args.algos:
        if a not in avail:
            raise ValueError(f"Algo '{a}' not available. Available: {sorted(avail)}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing logs to: {outdir.resolve()}")
    print(f"Functions: {args.funcs}")
    print(f"Dimensions: {args.dims}")
    print(f"Algorithms: {args.algos}")
    print(f"Runs per case: {args.runs}, Budget: {args.budget}")

    for func in args.funcs:
        for dim in args.dims:
            nd = normalize_dim_for_func(func, dim)
            for algo in args.algos:
                logfile = outdir / f"{func}_d{nd}_{algo}.csv"
                with logfile.open("w", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=[
                        "func","algo","dim","seed","best","evals","seconds","restarts",
                        "evals_to_target","success","params"
                    ])
                    writer.writeheader()
                    for r in range(args.runs):
                        seed = args.seed_offset + r
                        res = run_once(func, nd, algo, seed, args.budget)
                        writer.writerow(res)
                        print(f"{func:12s} d={nd:3d} algo={algo:10s} run={r:2d} "
                              f"best={res['best']:.3e} evals={res['evals']:6d} "
                              f"sec={res['seconds']:.2f} restarts={res['restarts']} "
                              f"succ={res['success']} et={res['evals_to_target']}")
                print(f"✅ Saved {logfile}")
    print("✅ Benchmark complete.")

if __name__ == "__main__":
    main()
