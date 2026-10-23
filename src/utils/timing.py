"""Per-stage wall-clock timing.

Runtime convention (see docs/DECISIONS.md):
  extract - computing the QUERY's features (never cached; this is online cost)
  search  - comparing the query against the database / producing the ranking
  verify  - geometric verification (SIFT+RANSAC, later hybrid reranking)
Database-side work (index) is timed once per run and reported separately.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

STAGES = ("extract", "search", "verify")


class StageTimer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        if name not in STAGES:
            raise ValueError(f"Unknown stage {name!r}; use one of {STAGES}")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - start
