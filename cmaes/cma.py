import warnings

from ._cma import CMA  # NOQA

warnings.warn(
    "This module is deprecated. Please import CMA class from the "
    "package root (ex: from cmaes import CMA).",
    FutureWarning,
)
