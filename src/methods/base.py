"""Retrieval-method contract (handoff section 11).

Every method (SIFT+RANSAC, SALAD, later the hybrids) implements this interface, and the runner
turns its outputs into a rankings file. The evaluator never imports a method.

Lifecycle:
    method = SomeMethod(config, cache_dir)
    method.index(split.references)          # offline: extract/cache database features
    result = method.retrieve(query, top_k)  # online: one query -> ranked reference ids
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..datasets.base import ImageRecord


@dataclass
class RetrievalResult:
    ranked: list[tuple[str, float]]  # (reference image_id, score), best first; higher score = better
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage, see utils/timing.py


class RetrievalMethod(ABC):
    name: str = "base"

    def __init__(self, config: dict, cache_dir: Path):
        self.config = config
        self.cache_dir = Path(cache_dir) / self.name  # each method caches under cache/<name>/
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def index(self, references: Sequence[ImageRecord]) -> dict:
        """Prepare the database (extract + cache features). Reuse cached features when present.

        Returns a small dict of stats for the run log, e.g. {"cached": 950, "extracted": 50}.
        """

    @abstractmethod
    def retrieve(self, query: ImageRecord, top_k: int) -> RetrievalResult:
        """Rank database images for one query. Query features must be computed here (not cached)
        and timed with StageTimer so runtime/query reflects online cost."""
