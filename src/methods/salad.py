"""Learned baseline: pretrained DINOv2 + SALAD global descriptors.  [Person 3]

Use the official pretrained model, do not retrain or modify it (handoff section 7):
    model = torch.hub.load(cfg["hub_repo"], cfg["hub_entry"]).eval()
(torch.hub imports SALAD's code, which needs pytorch-lightning, see requirements.txt. First call
downloads DINOv2 ViT-B/14 weights + dino_salad.ckpt into the torch hub cache.)

Preprocessing must match SALAD's eval.py: Resize(image_size, bilinear) -> ToTensor ->
Normalize(normalize_mean, normalize_std). Output is 8448-d (8192 + 256), already L2-normalized.

  index():    batch-encode all references, cache a float32 matrix + the ordered image_ids under
              self.cache_dir (key the cache file on the split's reference ids + config so a
              different split never silently reuses it).
  retrieve(): encode the query                         -> timer.stage("extract")
              scores = ref_matrix @ q (cosine)         -> timer.stage("search")
              return top_k by descending score

Device comes from cfg["device"] ("auto" = cuda if available). On CPU expect roughly seconds per
image, so cache the database once and reuse it.
"""
from __future__ import annotations

from typing import Sequence

from ..datasets.base import ImageRecord
from .base import RetrievalMethod, RetrievalResult


class SaladMethod(RetrievalMethod):
    name = "salad"

    def index(self, references: Sequence[ImageRecord]) -> dict:
        raise NotImplementedError("SALAD index(): see module docstring")

    def retrieve(self, query: ImageRecord, top_k: int) -> RetrievalResult:
        raise NotImplementedError("SALAD retrieve(): see module docstring")
