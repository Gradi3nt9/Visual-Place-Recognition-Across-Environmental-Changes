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

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..datasets.base import ImageRecord
from ..utils.timing import StageTimer
from .base import RetrievalMethod, RetrievalResult


class SaladMethod(RetrievalMethod):
    name = "salad"

    def _load_model(self) -> None:
        # Keep heavyweight dependencies lazy: the random baseline needs no torch/torchvision.
        if hasattr(self, "model"):
            return
        import torch
        from torchvision import transforms as T

        device = self.config["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in ("cpu", "cuda"):
            raise ValueError("SALAD device must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; set methods.salad.device=cpu")
        self.device = torch.device(device)
        self.transform = T.Compose([
            T.Resize(tuple(self.config["image_size"]), interpolation=T.InterpolationMode.BILINEAR,
                     antialias=True),
            T.ToTensor(),
            T.Normalize(self.config["normalize_mean"], self.config["normalize_std"]),
        ])
        self.model = torch.hub.load(self.config["hub_repo"], self.config["hub_entry"],
                                    pretrained=True).to(self.device).eval()

    @staticmethod
    def _validate_descriptors(matrix: np.ndarray, count: int) -> None:
        if matrix.shape != (count, 8448) or not np.isfinite(matrix).all():
            raise ValueError("SALAD descriptors must be finite with shape (N, 8448)")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
            raise ValueError("SALAD descriptors are not L2-normalized; cosine search is unsafe")

    def _encode(self, records: Sequence[ImageRecord]) -> np.ndarray:
        import torch
        from PIL import Image

        self._load_model()
        images = []
        for record in records:
            if record.path is None:
                raise ValueError(f"No resolved image path for {record.image_id}")
            try:
                with Image.open(record.path) as image:
                    images.append(self.transform(image.convert("RGB")))
            except OSError as exc:
                raise ValueError(f"Cannot read SALAD image {record.image_id}: {record.path}") from exc
        # Copying to CPU synchronizes CUDA before the extraction timer stops.
        with torch.inference_mode():
            matrix = self.model(torch.stack(images).to(self.device)).float().cpu().numpy()
        self._validate_descriptors(matrix, len(records))
        return matrix

    def index(self, references: Sequence[ImageRecord]) -> dict:
        if not references:
            raise ValueError("SALAD requires at least one reference image")
        if self.config["similarity"] != "cosine":
            raise ValueError("Only cosine SALAD search is supported")
        batch_size = self.config["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("SALAD batch_size must be a positive integer")
        self.ref_ids = [r.image_id for r in references]
        if len(set(self.ref_ids)) != len(self.ref_ids):
            raise ValueError("Duplicate SALAD reference IDs")
        # Include paths and file metadata to invalidate caches when the dataset changes.
        sources = []
        for record in references:
            if record.path is None:
                raise ValueError(f"No resolved image path for {record.image_id}")
            path = Path(record.path).resolve()
            stat = path.stat()
            sources.append((record.image_id, str(path), stat.st_size, stat.st_mtime_ns))
        signature = json.dumps({"version": 1, "config": self.config, "sources": sources},
                               sort_keys=True).encode()
        key = hashlib.sha256(signature).hexdigest()
        cache_path = self.cache_dir / f"references_{key}.npz"
        # Load the model during offline preparation even on a cache hit, so model startup
        # never contaminates the first query's extraction timing.
        self._load_model()
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                if cached["ids"].tolist() != self.ref_ids:
                    raise ValueError(f"SALAD cache reference IDs mismatch: {cache_path}")
                matrix = cached["descriptors"].astype(np.float32)
            self._validate_descriptors(matrix, len(references))
            cached_count = len(references)
        else:
            matrix = np.concatenate([
                self._encode(references[start:start + batch_size])
                for start in range(0, len(references), batch_size)
            ])
            np.savez(cache_path, ids=np.asarray(self.ref_ids), descriptors=matrix)
            cached_count = 0
        self.ref_matrix = matrix
        return {"references": len(references), "cached": cached_count,
                "extracted": len(references) - cached_count, "device": str(self.device),
                "precision": "float32", "descriptor_dim": matrix.shape[1],
                "cache_key": key}

    def retrieve(self, query: ImageRecord, top_k: int) -> RetrievalResult:
        if not hasattr(self, "ref_matrix"):
            raise RuntimeError("Call SALAD index() before retrieve()")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        timer = StageTimer()
        with timer.stage("extract"):
            descriptor = self._encode([query])[0]
        with timer.stage("search"):
            scores = self.ref_matrix @ descriptor
            order = np.argsort(-scores, kind="stable")[:top_k]
            ranked = [(self.ref_ids[i], float(scores[i])) for i in order]
        return RetrievalResult(ranked, timer.timings)
