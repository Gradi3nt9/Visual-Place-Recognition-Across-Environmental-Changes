"""Evaluator, rankings-file, and split tests on synthetic data (no images or downloads needed).
Run: python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.base import ImageRecord, VPRSplit, load_split, save_split  # noqa: E402
from src.datasets.nordland import build_split  # noqa: E402
from src.evaluation.recall import evaluate_recall  # noqa: E402
from src.evaluation.runtime import summarize_runtime  # noqa: E402
from src.utils.rankings import load_rankings, save_rankings  # noqa: E402


def _q(ranked_ids, timings=None):
    n = len(ranked_ids)
    return {"ranked": [[rid, float(n - i)] for i, rid in enumerate(ranked_ids)], "timings": timings or {}}


def test_recall_hand_computed():
    gt = {"q1": ["a"], "q2": ["b", "c"], "q3": ["d"]}
    queries = {
        "q1": _q(["a", "x", "y", "z", "w"]),   # rank 1
        "q2": _q(["x", "y", "c", "z", "w"]),   # rank 3 (any correct ref counts)
        "q3": _q(["x", "y", "z", "w", "v"]),   # miss
    }
    ev = evaluate_recall(queries, gt, [1, 5])
    assert ev["recall"] == {1: pytest.approx(1 / 3), 5: pytest.approx(2 / 3)}
    assert ev["per_query"] == {"q1": 1, "q2": 3, "q3": None}


def test_recall_rejects_incomplete_or_invalid_rankings():
    gt = {"q1": ["a"], "q2": ["b"]}
    with pytest.raises(ValueError, match="no ranking"):
        evaluate_recall({"q1": _q(["a"] * 5)}, gt, [1])
    with pytest.raises(ValueError, match="top_k"):
        evaluate_recall({"q1": _q(["a"]), "q2": _q(["b"])}, gt, [5])
    with pytest.raises(ValueError, match="not in the reference set"):
        evaluate_recall({"q1": _q(["a"]), "q2": _q(["zzz"])}, gt, [1], reference_ids={"a", "b"})


def test_runtime_summary():
    rt = summarize_runtime({"q1": _q(["a"], {"extract": 1.0, "search": 0.5}),
                            "q2": _q(["a"], {"extract": 3.0, "search": 0.5})})
    assert rt["extract"]["mean"] == 2.0 and rt["total"]["mean"] == 2.5 and "verify" not in rt


def test_rankings_roundtrip_and_sort_check(tmp_path):
    p = save_rankings(tmp_path / "r.json", {"method": "m"}, {"q1": _q(["a", "b"])})
    assert load_rankings(p)["queries"]["q1"]["ranked"][0][0] == "a"
    bad = save_rankings(tmp_path / "bad.json", {}, {"q1": {"ranked": [["a", 0.1], ["b", 0.9]]}})
    with pytest.raises(ValueError, match="sorted"):
        load_rankings(bad)


def test_split_validation_and_roundtrip(tmp_path):
    refs = [ImageRecord("r0", 0, "ref/0.jpg"), ImageRecord("r1", 1, "ref/1.jpg")]
    qs = [ImageRecord("q0", 0, "query/0.jpg")]
    split = VPRSplit("toy", refs, qs, {"q0": ["r1"]}, {"note": "toy"})
    save_split(split, tmp_path)
    loaded = load_split("toy", data_root=tmp_path, splits_dir=tmp_path)
    assert loaded.ground_truth == {"q0": ["r1"]} and loaded.references[1].path == tmp_path / "ref/1.jpg"
    with pytest.raises(ValueError):
        VPRSplit("bad", refs, qs, {"q0": ["nope"]}).validate()


@pytest.fixture
def fake_nordland_gt(tmp_path):
    """Synthetic files with the official structure: query i <-> refs {10i-1, 10i, 10i+1}."""
    n_q, n_ref = 30, 300
    np.save(tmp_path / "Nordland_dbImages.npy", np.array([f"ref/{i:07d}.jpg" for i in range(n_ref)]))
    np.save(tmp_path / "Nordland_qImages.npy", np.array([f"query/{i:07d}.jpg" for i in range(n_q)]))
    gt = np.empty(n_q, dtype=object)
    for i in range(n_q):
        gt[i] = np.array([j for j in (10 * i - 1, 10 * i, 10 * i + 1) if 0 <= j < n_ref])
    np.save(tmp_path / "Nordland_gt.npy", gt, allow_pickle=True)
    return tmp_path


def test_nordland_subset_preserves_ground_truth(fake_nordland_gt):
    full = build_split(fake_nordland_gt, "full")
    assert full.ground_truth["nordland_query_0000005"] == [
        "nordland_ref_0000049", "nordland_ref_0000050", "nordland_ref_0000051"]

    sub = build_split(fake_nordland_gt, "sub", query_start=10, num_queries=5, ref_stride=10)
    sub.validate()
    assert len(sub.queries) == 5
    # with stride 10 only the exact frame 10i survives as the correct reference
    assert sub.ground_truth["nordland_query_0000012"] == ["nordland_ref_0000120"]
    assert all(r.frame_index % 10 == 0 for r in sub.references)

    # stride that removes every correct ref for some queries -> those queries are dropped, not mislabeled
    odd = build_split(fake_nordland_gt, "odd", query_start=10, num_queries=5, ref_stride=7)
    odd.validate()
    assert odd.meta["counts"]["queries"] + odd.meta["counts"]["queries_dropped_no_gt"] == 5
