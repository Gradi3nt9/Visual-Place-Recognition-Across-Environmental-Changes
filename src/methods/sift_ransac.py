"""Classical baseline: SIFT + kNN matching + Lowe ratio + fundamental-matrix RANSAC.  [Person 2]

Pipeline per handoff sections 8-10 (parameters come from configs/default.yaml, methods.sift_ransac):
  index():    for each reference, run cv2.SIFT_create() once and cache keypoint xy + descriptors
              under self.cache_dir (e.g. one .npz per image_id). Skip images already cached.
  retrieve(): extract query SIFT                                   -> timer.stage("extract")
              for each reference: BFMatcher knnMatch(k=knn_k),
                  Lowe ratio test (ratio_test),
                  cv2.findFundamentalMat(FM_RANSAC, ransac_reproj_threshold, ransac_confidence),
                  score = inlier count                             -> timer.stage("verify")
              sort references by score, return top_k               -> timer.stage("search")

Failure cases must yield score 0, never crash the run: missing/unreadable image, no descriptors,
fewer than min_matches matches, findFundamentalMat returning None.

Keep the score function separate (e.g. a `score_from_inliers(mask, matches)` helper) so it can later
become inlier ratio / normalized count without touching the matching code. The hybrid methods will
reuse the pairwise `verify(query_feats, ref_feats)` step, so expose it as its own function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm

from ..datasets.base import ImageRecord
from ..utils.timing import StageTimer
from .base import RetrievalMethod, RetrievalResult


def extract_sift(
    image_path: str | Path | None,
    sift: cv2.SIFT | None = None,
    max_image_side: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract SIFT keypoint coordinates and descriptors from an image.

    Returns:
        (xy, descs): xy is (N, 2) float32 coordinates, descs is (N, 128) float32 descriptors.
        If the image is unreadable or has no keypoints, returns empty (0, 2) and (0, 128) arrays.
    """
    empty_xy = np.empty((0, 2), dtype=np.float32)
    empty_descs = np.empty((0, 128), dtype=np.float32)

    if image_path is None:
        return empty_xy, empty_descs

    path_str = str(image_path)
    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return empty_xy, empty_descs

    h, w = img.shape[:2]
    if max_image_side is not None and max(h, w) > max_image_side:
        scale = max_image_side / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if sift is None:
        sift = cv2.SIFT_create()

    kps, descs = sift.detectAndCompute(img, None)
    if descs is None or len(kps) == 0:
        return empty_xy, empty_descs

    xy = np.array([kp.pt for kp in kps], dtype=np.float32)
    descs = descs.astype(np.float32)
    return xy, descs


def score_from_inliers(
    mask: np.ndarray | None,
    num_matches: int,
    score_type: str = "inlier_count",
) -> float:
    """Compute geometric score from the RANSAC inlier mask.

    Modular scoring function: supports inlier_count and inlier_ratio.
    """
    if mask is None:
        inliers = 0
    else:
        inliers = int(np.sum(mask.ravel() == 1))

    if score_type == "inlier_count":
        return float(inliers)
    elif score_type == "inlier_ratio":
        return float(inliers / num_matches) if num_matches > 0 else 0.0
    else:
        raise ValueError(f"Unknown geometric score type: {score_type!r}")


def verify_pair(
    query_feats: tuple[np.ndarray, np.ndarray],
    ref_feats: tuple[np.ndarray, np.ndarray],
    config: dict,
    matcher: cv2.BFMatcher | None = None,
) -> tuple[float, int, int]:
    """Verify a query-reference pair using kNN, Lowe ratio test, and Fundamental matrix RANSAC.

    Parameters:
        query_feats: (xy, descs) tuple for the query image.
        ref_feats: (xy, descs) tuple for the reference image.
        config: method config dict containing ratio_test, min_matches,
                ransac_reproj_threshold, ransac_confidence, score.
        matcher: optional cv2.BFMatcher instance to reuse across pairs.

    Returns:
        (score, inlier_count, num_matches): score float, RANSAC inliers count,
        and count of Lowe-filtered matches.
    """
    q_xy, q_descs = query_feats
    ref_xy, ref_descs = ref_feats

    min_matches = config.get("min_matches", 8)
    knn_k = config.get("knn_k", 2)
    ratio_test = config.get("ratio_test", 0.75)
    reproj_thresh = config.get("ransac_reproj_threshold", 3.0)
    confidence = config.get("ransac_confidence", 0.99)
    score_type = config.get("score", "inlier_count")

    # Fast-path failure check: fundamental matrix estimation requires at least 8 matches
    if len(q_descs) < min_matches or len(ref_descs) < min_matches:
        return 0.0, 0, 0

    if matcher is None:
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    # 1. k-NN matching
    raw_matches = matcher.knnMatch(q_descs, ref_descs, k=knn_k)

    # 2. Lowe ratio test
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_test * n.distance:
                good_matches.append(m)

    num_good = len(good_matches)
    if num_good < min_matches:
        return 0.0, 0, num_good

    pts_q = q_xy[[m.queryIdx for m in good_matches]]
    pts_ref = ref_xy[[m.trainIdx for m in good_matches]]

    # 3. Fundamental matrix estimation with RANSAC
    try:
        F, mask = cv2.findFundamentalMat(
            pts_q,
            pts_ref,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=reproj_thresh,
            confidence=confidence,
        )
    except cv2.error:
        return 0.0, 0, num_good

    if F is None or mask is None:
        return 0.0, 0, num_good

    inliers = int(np.sum(mask.ravel() == 1))
    score = score_from_inliers(mask, num_good, score_type=score_type)
    return score, inliers, num_good


class SiftRansacMethod(RetrievalMethod):
    name = "sift_ransac"

    def __init__(self, config: dict, cache_dir: Path):
        super().__init__(config, cache_dir)
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.max_image_side = self.config.get("max_image_side")
        self.ref_records: list[ImageRecord] = []
        self.ref_features: list[tuple[np.ndarray, np.ndarray]] = []

    def _cache_path(self, image_id: str) -> Path:
        tag = f"_side{self.max_image_side}" if self.max_image_side else ""
        return self.cache_dir / f"{image_id}{tag}.npz"

    def _extract(self, path: Path | str | None) -> tuple[np.ndarray, np.ndarray]:
        return extract_sift(path, sift=self.sift, max_image_side=self.max_image_side)

    def index(self, references: Sequence[ImageRecord]) -> dict:
        """Extract and cache reference features offline.

        Features are saved as .npz under self.cache_dir and retained in memory
        for fast pairwise evaluation during query retrieval.
        """
        if not references:
            raise ValueError("SIFT index() requires at least one reference image")

        self.ref_records = list(references)
        self.ref_features = []
        cached_count = 0
        extracted_count = 0

        use_pbar = len(references) > 100
        iterator = tqdm(references, desc="Indexing SIFT references") if use_pbar else references

        for r in iterator:
            cache_path = self._cache_path(r.image_id)
            if cache_path.exists():
                try:
                    with np.load(cache_path) as data:
                        xy = data["xy"]
                        descs = data["descs"]
                    cached_count += 1
                except Exception:
                    xy, descs = self._extract(r.path)
                    np.savez(cache_path, xy=xy, descs=descs)
                    extracted_count += 1
            else:
                xy, descs = self._extract(r.path)
                np.savez(cache_path, xy=xy, descs=descs)
                extracted_count += 1

            self.ref_features.append((xy, descs))

        return {
            "references": len(references),
            "cached": cached_count,
            "extracted": extracted_count,
        }

    def retrieve(self, query: ImageRecord, top_k: int) -> RetrievalResult:
        """Online query retrieval with stage timing per docs/DECISIONS.md:

        extract: compute query SIFT keypoints and descriptors (never cached)
        verify:  pairwise kNN + Lowe ratio + F-matrix RANSAC against database
        search:  sort references descending by score, slice top_k
        """
        if not self.ref_records or not self.ref_features:
            raise RuntimeError("Must call SiftRansacMethod.index() before retrieve()")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        timer = StageTimer()

        # 1. Online query feature extraction (never cached)
        with timer.stage("extract"):
            q_feats = self._extract(query.path)

        # 2. Pairwise geometric verification
        scores: list[float] = []
        with timer.stage("verify"):
            for ref_feats in self.ref_features:
                score, _, _ = verify_pair(
                    q_feats,
                    ref_feats,
                    self.config,
                    matcher=self.matcher,
                )
                scores.append(score)

        # 3. Candidate ranking (higher score = better)
        with timer.stage("search"):
            scores_arr = np.asarray(scores, dtype=np.float32)
            order = np.argsort(-scores_arr, kind="stable")[:top_k]
            ranked = [(self.ref_records[i].image_id, float(scores_arr[i])) for i in order]

        return RetrievalResult(ranked, timer.timings)
