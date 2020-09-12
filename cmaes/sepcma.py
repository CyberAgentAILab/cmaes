import warnings

from ._sepcma import SepCMA  # NOQA

warnings.warn(
    "This module is deprecated. Please import SepCMA class from the "
    "package root (ex: from cmaes import SepCMA).",
    FutureWarning,
)
