"""SALAD contract tests without model weights, torch, or dataset downloads."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.base import ImageRecord
from src.methods.salad import SaladMethod
from src.utils.config import load_config


def test_cache_and_uncached_query_search(tmp_path, monkeypatch):
    cfg = load_config()["methods"]["salad"]
    records = []
    for i in range(3):
        path = tmp_path / f"{i}.jpg"
        path.write_bytes(b"placeholder")
        records.append(ImageRecord(f"r{i}", i, path.name, path))
    calls = []

    def fake_load(self):
        self.device = "cpu"

    def fake_encode(self, items):
        calls.extend(r.image_id for r in items)
        matrix = np.zeros((len(items), 8448), dtype=np.float32)
        for i, record in enumerate(items):
            matrix[i, record.frame_index] = 1
        return matrix

    monkeypatch.setattr(SaladMethod, "_load_model", fake_load)
    monkeypatch.setattr(SaladMethod, "_encode", fake_encode)
    method = SaladMethod(cfg, tmp_path / "cache")
    assert method.index(records)["extracted"] == 3
    calls.clear()
    second = SaladMethod(cfg, tmp_path / "cache")
    assert second.index(records)["cached"] == 3
    assert calls == []
    query = ImageRecord("q", 1, records[1].rel_path, records[1].path)
    for _ in range(2):
        result = second.retrieve(query, 2)
        assert result.ranked == [("r1", 1.0), ("r0", 0.0)]
        assert set(result.timings) == {"extract", "search"}
    assert calls == ["q", "q"]
    calls.clear()
    records[0].path.write_bytes(b"changed image")
    assert second.index(records)["extracted"] == 3
    assert calls == ["r0", "r1", "r2"]


def test_reject_invalid_descriptors():
    matrix = np.zeros((1, 8448), dtype=np.float32)
    with pytest.raises(ValueError, match="L2-normalized"):
        SaladMethod._validate_descriptors(matrix, 1)
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SaladMethod._validate_descriptors(matrix, 1)
