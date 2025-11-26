# benchmark/sensitivity.py
import argparse, itertools, csv
from pathlib import Path
import numpy as np
from cmaes import IPOPSepCMA
from benchmark.functions import FUNCS
import time

parser = argparse.ArgumentParser()
parser.add_argument("--func", default="rastrigin")
parser.add_argument("--dim", type=int, default=20)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--budget", type=int, default=20000)
parser.add_argument("--out", default="benchmark/logs/sensitivity.csv")
args = parser.parse_args()

dim = args.dim
f = FUNCS[args.func]
bounds = None
if args.func in {"rastrigin","ackley"}:
    b = np.empty((dim,2), float); b[:,0], b[:,1] = -5,5; bounds=b

PAT = [8, 12, 20]
INC = [2, 3]
MSIG = [1e-10, 1e-8]
grid = list(itertools.product(PAT, INC, MSIG))

Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["patience","inc_popsize","min_sigma","seed","best","evals","seconds","restarts"])
    writer.writeheader()
    for (pat, inc, msig) in grid:
        for seed in range(args.runs):
            opt = IPOPSepCMA(mean=np.ones(dim)*3.0, sigma=2.0, seed=seed, bounds=bounds,
                             patience=pat, inc_popsize=inc, min_sigma=msig)
            evals=0; best=np.inf; t0=time.time()
            while True:
                batch=[]
                for _ in range(opt.population_size):
                    x=opt.ask(); fx=float(f(x)); batch.append((x,fx)); evals+=1; best=min(best,fx)
                    if evals>=args.budget: break
                if len(batch)<opt.population_size: break
                opt.tell(batch)
                if evals>=args.budget or getattr(opt,"should_stop",lambda:False)(): break
            writer.writerow(dict(patience=pat,inc_popsize=inc,min_sigma=msig,seed=seed,best=best,evals=evals,seconds=time.time()-t0,restarts=getattr(opt,"restart_count",0)))
print("✅ sensitivity.csv saved.")
