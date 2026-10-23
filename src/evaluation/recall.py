"""Recall@K: a query counts as correct at K if ANY of its correct references is in the top K.
Same definition as SALAD's get_validation_recalls."""
from __future__ import annotations


def first_correct_rank(ranked_ids: list[str], correct: set[str]) -> int | None:
    """1-based rank of the first correct reference, or None if none is retrieved."""
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in correct:
            return rank
    return None


def evaluate_recall(queries: dict[str, dict], ground_truth: dict[str, list[str]],
                    ks: list[int], reference_ids: set[str] | None = None) -> dict:
    """Compute Recall@K over every query in ground_truth.

    queries:       rankings-file "queries" section {qid: {"ranked": [[ref_id, score], ...], ...}}
    ground_truth:  {qid: [correct ref ids]}
    reference_ids: if given, rankings containing ids outside the database raise an error.
    Returns {"n_queries", "recall": {k: value}, "per_query": {qid: first_correct_rank}}.
    """
    missing = set(ground_truth) - set(queries)
    if missing:
        raise ValueError(f"{len(missing)} queries have no ranking, e.g. {sorted(missing)[:3]}")
    extra = set(queries) - set(ground_truth)
    if extra:
        raise ValueError(f"{len(extra)} ranked queries are not in the split, e.g. {sorted(extra)[:3]}")

    max_len = min(len(e["ranked"]) for e in queries.values()) if queries else 0
    too_short = [k for k in ks if k > max_len]
    if too_short:
        raise ValueError(f"Rankings were saved with top_k={max_len}; cannot compute Recall@{too_short}")

    per_query = {}
    for qid, correct in ground_truth.items():
        ranked_ids = [rid for rid, _ in queries[qid]["ranked"]]
        if reference_ids is not None and not set(ranked_ids) <= reference_ids:
            raise ValueError(f"Ranking for {qid} contains ids not in the reference set")
        per_query[qid] = first_correct_rank(ranked_ids, set(correct))

    n = len(per_query)
    recall = {k: sum(1 for r in per_query.values() if r is not None and r <= k) / n for k in ks}
    return {"n_queries": n, "recall": recall, "per_query": per_query}
