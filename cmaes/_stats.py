import math
import numpy as np


@np.vectorize
def norm_cdf(x: float, loc: float = 0.0, scale: float = 1.0) -> float:
    x = (x - loc) / scale
    x = x / 2**0.5
    z = abs(x)

    if z < 1 / 2**0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 0.5 * math.erfc(z)
        if x > 0:
            y = 1.0 - y

    return y


@np.vectorize
def chi2_ppf(q: float) -> float:
    """
    only deal with the special case df=1, loc=0, scale=1
    solve chi2.cdf(x; df=1) = erf(sqrt(x/2)) = q with bisection method
    """
    if q == 0:
        return 0.0
    if q == 1:
        return math.inf
    a, b = 0.0, 100.0
    if q < 0.9:
        for _ in range(100):
            m = (a + b) / 2
            if math.erf(math.sqrt(m / 2)) < q:
                a = m
            else:
                b = m
    else:
        for _ in range(100):
            m = (a + b) / 2
            if math.erfc(math.sqrt(m / 2)) > 1.0 - q:
                a = m
            else:
                b = m
    return m
