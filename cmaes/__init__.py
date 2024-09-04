from ._cma import CMA  # NOQA
from ._sepcma import SepCMA  # NOQA
from ._warm_start import get_warm_start_mgd  # NOQA
from ._cmawm import CMAwM  # NOQA
from ._xnes import XNES  # NOQA
from ._dxnesic import DXNESIC  # NOQA
from ._catcma import CatCMA  # NOQA

try:
    from ._safe_cma import SafeCMA  # NOQA
except ImportError:
    pass  # Implementation of Safe CMA-ES requires scipy, gpytorch, and torch

__version__ = "0.11.1"
