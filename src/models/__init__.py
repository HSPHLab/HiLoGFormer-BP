"""Importing this package registers every model."""
from .registry import MODELS, build_model, register_model  # noqa: F401
from . import resnet1d  # noqa: F401
try:
    from . import hicaf_bp  # noqa: F401
    from . import hicaf_bp_cal  # noqa: F401
except ImportError as e:  # pragma: no cover
    if "hicaf_bp" not in str(e):
        raise
