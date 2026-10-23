"""Nordland in the VPR-Bench format used by SALAD's official evaluation.

Established protocol (inspected from serizba/salad, datasets/Nordland/*.npy):
  - 27,592 reference images  ref/0000000.jpg ... ref/0027591.jpg
  - 2,760 query images       query/0000000.jpg ... query/0002759.jpg
  - query i is correct for reference frames {10i-1, 10i, 10i+1} (clipped to the valid range),
    i.e. queries are every 10th frame and the tolerance is +/-1 reference frame.
We build our ground truth from those official files rather than inventing a threshold.

Subsets keep the ground truth consistent: references are chosen first, then each query's correct
set is intersected with the kept references, and queries left with no correct reference are dropped
(and counted in meta.json).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .base import ImageRecord, VPRSplit

# Official files, pinned to a specific commit of the SALAD repository.
SALAD_COMMIT = "6aede13a3f6c25750bf7fde10209c06cb73060bb"
GT_URL = f"https://raw.githubusercontent.com/serizba/salad/{SALAD_COMMIT}/datasets/Nordland/" + "{name}"
GT_FILES = ("Nordland_dbImages.npy", "Nordland_qImages.npy", "Nordland_gt.npy")


def _frame(rel_path: str) -> int:
    return int(re.search(r"(\d+)\.\w+$", rel_path).group(1))


def load_official_gt(gt_dir: Path) -> tuple[list[str], list[str], list[list[int]]]:
    """Return (reference rel_paths, query rel_paths, per-query lists of correct reference indices)."""
    db = [str(p) for p in np.load(gt_dir / "Nordland_dbImages.npy")]
    q = [str(p) for p in np.load(gt_dir / "Nordland_qImages.npy")]
    gt = [sorted(int(i) for i in g) for g in np.load(gt_dir / "Nordland_gt.npy", allow_pickle=True)]
    if len(gt) != len(q):
        raise ValueError("Nordland ground truth length does not match number of queries")
    return db, q, gt


def build_split(gt_dir: Path, name: str, query_start: int = 0, num_queries: int | None = None,
                ref_stride: int = 1, ref_margin: int = 0) -> VPRSplit:
    """Build a (sub)split.

    query_start / num_queries : contiguous block of official queries (None = all from start).
    ref_stride                : keep only reference frames whose index is a multiple of this.
    ref_margin                : extra reference frames on each side of the queries' ground-truth span
                                (distractors). The database is the contiguous span, then strided.
    """
    db, q, gt = load_official_gt(gt_dir)
    q_end = len(q) if num_queries is None else min(len(q), query_start + num_queries)
    q_sel = range(query_start, q_end)
    if not q_sel:
        raise ValueError("query selection is empty")
    span = [i for qi in q_sel for i in gt[qi]]
    lo, hi = max(0, min(span) - ref_margin), min(len(db) - 1, max(span) + ref_margin)
    ref_idx = [i for i in range(lo, hi + 1) if i % ref_stride == 0]
    kept = set(ref_idx)

    refs = [ImageRecord(f"nordland_ref_{_frame(db[i]):07d}", _frame(db[i]), db[i]) for i in ref_idx]
    ref_id = {i: r.image_id for i, r in zip(ref_idx, refs)}
    queries, ground_truth, dropped = [], {}, 0
    for qi in q_sel:
        correct = [ref_id[i] for i in gt[qi] if i in kept]
        if not correct:
            dropped += 1
            continue
        rec = ImageRecord(f"nordland_query_{_frame(q[qi]):07d}", _frame(q[qi]), q[qi])
        queries.append(rec)
        ground_truth[rec.image_id] = correct

    meta = {
        "dataset": "Nordland (VPR-Bench format, as used by SALAD eval)",
        "seasons": {"reference": "summer (TO VERIFY visually)", "query": "winter (TO VERIFY visually)"},
        "gt_source": GT_URL.format(name="*.npy"),
        "gt_protocol": "query i <-> reference frames {10i-1, 10i, 10i+1} from official Nordland_gt.npy",
        "params": {"query_start": query_start, "num_queries": num_queries,
                   "ref_stride": ref_stride, "ref_margin": ref_margin},
        "counts": {"references": len(refs), "queries": len(queries),
                   "queries_dropped_no_gt": dropped,
                   "official_references": len(db), "official_queries": len(q)},
    }
    return VPRSplit(name, refs, queries, ground_truth, meta)
