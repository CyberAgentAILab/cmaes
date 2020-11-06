import warnings

from ._cma import CMA

__all__ = ["CMA"]

warnings.warn(
    "This module is deprecated. Please import CMA class from the "
    "package root (ex: from cmaes import CMA).",
    FutureWarning,
)
