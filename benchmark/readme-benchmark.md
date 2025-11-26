# Extended CMA-ES Restart Benchmarks

This branch adds an extended benchmark setup and analysis dashboard on top of the original `cmaes` library.

## Algorithms

We benchmark the following CMA-ES variants:

- `CMA` (full covariance) – baseline
- `SepCMA` (diagonal covariance)
- `IPOP-Full` and `BIPOP-Full` – restart strategies on full covariance CMA-ES
- `IPOP-Sep` and `BIPOP-Sep` – restart strategies on SepCMA (diagonal)

The restart wrappers follow the ideas of Auger & Hansen (2005) for IPOP/BIPOP with an ask–tell API consistent with the rest of the library.

## Benchmark design

- Functions: Sphere, Ellipsoid, Rastrigin, Ackley, Rosenbrock  
- Dimensions: 10, 20, 50, 100  
- Budget: up to 500,000 function evaluations per run  
- Repetitions: 20 independent runs per (function, dimension, algorithm)

The script `benchmark/bench_hybrid.py` runs all combinations and logs one CSV per configuration under `benchmark/logs/`.

Example (small) run:

```bash
python -m benchmark.bench_hybrid \
  --funcs sphere,rastrigin \
  --dims 10,20 \
  --algos cma_full,sep,ipop_sep,bipop_sep \
  --runs 5 \
  --budget 20000
