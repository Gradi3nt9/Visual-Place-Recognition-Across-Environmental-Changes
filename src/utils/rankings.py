"""The rankings file: the ONLY thing the evaluator reads from a method.

Format (JSON):
{
  "format_version": 1,
  "meta":    {method, method_config, split, top_k, index_seconds, git_commit, ...},
  "queries": {query_id: {"ranked": [[ref_id, score], ...],   # best first
                         "timings": {"extract": s, "search": s, "verify": s}}}
}
Higher score = better match, for every method.
"""
from __future__ import annotations

import json
from pathlib import Path

FORMAT_VERSION = 1


def save_rankings(path: str | Path, meta: dict, queries: dict[str, dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, "meta": meta, "queries": queries}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def load_rankings(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported rankings format_version {payload.get('format_version')}")
    for qid, entry in payload["queries"].items():
        scores = [s for _, s in entry["ranked"]]
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError(f"{path}: ranking for {qid} is not sorted best-first")
    return payload
