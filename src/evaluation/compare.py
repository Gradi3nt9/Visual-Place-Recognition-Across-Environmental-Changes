"""Checks that rows placed in one comparison table are fair to compare (handoff §10, DECISIONS §4)."""
from __future__ import annotations


def check_comparable(rows: list[dict]) -> list[str]:
    """Raise if rows use different splits or query counts; return warnings for different machines."""
    if len(rows) < 2:
        return []
    splits = {r["split"] for r in rows}
    if len(splits) > 1:
        raise ValueError(f"rows use different splits: {sorted(splits)}")
    counts = {r["queries"] for r in rows}
    if len(counts) > 1:
        raise ValueError(f"rows evaluate different numbers of queries: {sorted(counts)}")
    machines = {r["machine"] for r in rows}
    if len(machines) > 1:
        return [f"rows come from different machines ({'; '.join(sorted(machines))}); "
                "runtime columns are not directly comparable"]
    return []
