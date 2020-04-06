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

    # TODO(c-bata): Set the upper version constraint after merged to Optuna.
    # https://github.com/optuna/optuna/pull/885
    if get_optuna_version() < (1, 3):
        return

    from .sampler import _fast_intersection_search_space

    optuna.samplers.intersection_search_space = _fast_intersection_search_space
