"""Random ranking. Exists to test the pipeline end to end before real methods work, and as a floor:
expected Recall@1 is roughly (#correct refs per query) / (#references)."""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np

from ..datasets.base import ImageRecord
from ..utils.timing import StageTimer
from .base import RetrievalMethod, RetrievalResult


class RandomMethod(RetrievalMethod):
    name = "random"

    def index(self, references: Sequence[ImageRecord]) -> dict:
        self.ref_ids = [r.image_id for r in references]
        return {"references": len(self.ref_ids)}

    def retrieve(self, query: ImageRecord, top_k: int) -> RetrievalResult:
        timer = StageTimer()
        with timer.stage("search"):
            # Seed per query so results do not depend on query order.
            rng = np.random.default_rng(zlib.crc32(query.image_id.encode()) + self.config.get("seed", 0))
            scores = rng.random(len(self.ref_ids))
            order = np.argsort(-scores)[:top_k]
        return RetrievalResult([(self.ref_ids[i], float(scores[i])) for i in order], timer.timings)
