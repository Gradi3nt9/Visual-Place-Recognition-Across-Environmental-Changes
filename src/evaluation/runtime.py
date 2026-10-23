"""Summarize per-query stage timings recorded in a rankings file."""
from __future__ import annotations

import statistics

from ..utils.timing import STAGES


def summarize_runtime(queries: dict[str, dict]) -> dict:
    """Mean/median seconds per query, per stage and in total. Stages a method never uses are omitted."""
    totals = [sum(e.get("timings", {}).values()) for e in queries.values()]
    out = {}
    for stage in STAGES:
        vals = [e["timings"][stage] for e in queries.values() if stage in e.get("timings", {})]
        if vals:
            out[stage] = {"mean": statistics.fmean(vals), "median": statistics.median(vals)}
    if totals:
        out["total"] = {"mean": statistics.fmean(totals), "median": statistics.median(totals)}
    return out
