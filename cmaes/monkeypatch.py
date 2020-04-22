import warnings
from typing import Tuple

try:
    import optuna
except:  # noqa: E722
    optuna = None


def get_optuna_version() -> Tuple[int, int]:
    optuna_ver_info = optuna.__version__.split(".")
    major_ver, minor_ver = int(optuna_ver_info[0]), int(optuna_ver_info[1])
    return major_ver, minor_ver


def patch_fast_intersection_search_space() -> None:
    if optuna is None:
        return

    # CMA-ES sampler is added at v1.3.0.
    # https://github.com/optuna/optuna/pull/1142 is added at v1.4.0.
    if get_optuna_version() != (1, 3):
        warnings.warn(
            "intersection_search_space will be faster than v1.4.0. "
            "You don't need to apply this monkeypatch",
            DeprecationWarning,
            stacklevel=2,
        )
        return

    from .sampler import _fast_intersection_search_space

    optuna.samplers.intersection_search_space = _fast_intersection_search_space
