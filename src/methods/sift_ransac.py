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

from typing import Sequence

from ..datasets.base import ImageRecord
from .base import RetrievalMethod, RetrievalResult


class SiftRansacMethod(RetrievalMethod):
    name = "sift_ransac"

    def index(self, references: Sequence[ImageRecord]) -> dict:
        raise NotImplementedError("SIFT index(): see module docstring")

    def retrieve(self, query: ImageRecord, top_k: int) -> RetrievalResult:
        raise NotImplementedError("SIFT retrieve(): see module docstring")
