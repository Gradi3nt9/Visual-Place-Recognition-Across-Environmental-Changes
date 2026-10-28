"""Unit tests for the SIFT + RANSAC classical baseline (Person 2)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.datasets.base import ImageRecord
from src.methods.sift_ransac import (
    SiftRansacMethod,
    extract_sift,
    score_from_inliers,
    verify_pair,
)


def test_score_from_inliers():
    mask = np.array([[1], [0], [1], [1], [0]], dtype=np.uint8)
    assert score_from_inliers(mask, 5, "inlier_count") == 3.0
    assert score_from_inliers(mask, 5, "inlier_ratio") == pytest.approx(0.6)
    assert score_from_inliers(None, 0, "inlier_count") == 0.0
    assert score_from_inliers(None, 0, "inlier_ratio") == 0.0

    with pytest.raises(ValueError, match="Unknown geometric score type"):
        score_from_inliers(mask, 5, "invalid_type")


def test_extract_sift_missing_or_corrupt_file(tmp_path):
    xy, descs = extract_sift(tmp_path / "nonexistent.jpg")
    assert xy.shape == (0, 2)
    assert descs.shape == (0, 128)

    # Corrupt empty file
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_text("not an image")
    xy2, descs2 = extract_sift(bad_file)
    assert xy2.shape == (0, 2)
    assert descs2.shape == (0, 128)


def test_verify_pair_few_matches():
    cfg = {"min_matches": 8, "knn_k": 2, "ratio_test": 0.75}
    q_feats = (np.zeros((3, 2), dtype=np.float32), np.zeros((3, 128), dtype=np.float32))
    ref_feats = (np.zeros((3, 2), dtype=np.float32), np.zeros((3, 128), dtype=np.float32))

    score, inliers, matches = verify_pair(q_feats, ref_feats, cfg)
    assert score == 0.0
    assert inliers == 0
    assert matches == 0


def test_sift_ransac_index_and_retrieve_end_to_end(tmp_path):
    # Create two synthetic distinct test images with textured shapes
    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"

    canvas1 = np.zeros((200, 200), dtype=np.uint8)
    # Add high-contrast geometric patterns for SIFT features
    for i in range(10, 190, 20):
        cv2.rectangle(canvas1, (i, 10), (i + 10, 190), 255, -1)
    cv2.imwrite(str(img1_path), canvas1)

    canvas2 = np.zeros((200, 200), dtype=np.uint8)
    for j in range(10, 190, 20):
        cv2.circle(canvas2, (j, j), 8, 255, -1)
    cv2.imwrite(str(img2_path), canvas2)

    rec1 = ImageRecord("ref_01", 1, "ref/01.jpg", img1_path)
    rec2 = ImageRecord("ref_02", 2, "ref/02.jpg", img2_path)
    q_rec = ImageRecord("query_01", 1, "query/01.jpg", img1_path)  # Matches rec1 exactly

    cfg = {
        "knn_k": 2,
        "ratio_test": 0.75,
        "min_matches": 8,
        "ransac_reproj_threshold": 3.0,
        "ransac_confidence": 0.99,
        "score": "inlier_count",
        "max_image_side": None,
    }

    cache_dir = tmp_path / "cache"
    method = SiftRansacMethod(cfg, cache_dir)

    # First index: should extract
    stats = method.index([rec1, rec2])
    assert stats["references"] == 2
    assert stats["extracted"] == 2
    assert stats["cached"] == 0

    # Second index: should use cache
    method2 = SiftRansacMethod(cfg, cache_dir)
    stats2 = method2.index([rec1, rec2])
    assert stats2["cached"] == 2
    assert stats2["extracted"] == 0

    # Retrieve
    res = method2.retrieve(q_rec, top_k=2)
    assert len(res.ranked) == 2
    top_id, top_score = res.ranked[0]
    assert top_id == "ref_01"
    assert top_score > 0
    assert set(res.timings.keys()) == {"extract", "verify", "search"}
