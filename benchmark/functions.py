# benchmark/functions.py
from __future__ import annotations
import numpy as np

def sphere(x: np.ndarray) -> float:
    # mean of squares (matches your kurobako Sphere)
    return float(np.mean(x**2))

def rastrigin(x: np.ndarray) -> float:
    d = x.size
    return float(10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def rosenbrock(x: np.ndarray) -> float:
    # classic 2D form generalised to nD (sum over pairs)
    return float(np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2))

def himmelblau(x: np.ndarray) -> float:
    # defined for 2D; if higher-d, evaluate on first two dims
    x1, x2 = float(x[0]), float(x[1])
    return (x1**2 + x2 - 11.0)**2 + (x1 + x2**2 - 7.0)**2

def six_hump_camel(x: np.ndarray) -> float:
    # defined for 2D
    x1, x2 = float(x[0]), float(x[1])
    return (4 - 2.1*(x1**2) + (x1**4)/3.0)*(x1**2) + x1*x2 + (-4 + 4*(x2**2))*(x2**2)

def ellipsoid(x: np.ndarray) -> float:
    # separable ill-conditioned (weights grow geometrically)
    n = x.size
    if n < 2:
        raise ValueError("dimension must be >= 2")
    weights = 1000.0 ** (np.arange(n, dtype=float) / (n - 1))
    return float(np.sum((weights * x)**2))

def ackley(x: np.ndarray) -> float:
    # standard Ackley (a=20, b=0.2, c=2π)
    a, b, c = 20.0, 0.2, 2.0*np.pi
    d = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c*x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    return float(term1 + term2 + a + np.e)

# registry for convenience
FUNCS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "himmelblau": himmelblau,
    "six_hump_camel": six_hump_camel,
    "ellipsoid": ellipsoid,
    "ackley": ackley,
}
