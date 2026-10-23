"""Method registry: the runner looks methods up by the name used in the config."""
from .base import RetrievalMethod, RetrievalResult
from .random_baseline import RandomMethod
from .salad import SaladMethod
from .sift_ransac import SiftRansacMethod

METHODS: dict[str, type[RetrievalMethod]] = {
    m.name: m for m in (RandomMethod, SiftRansacMethod, SaladMethod)
}

__all__ = ["METHODS", "RetrievalMethod", "RetrievalResult"]
