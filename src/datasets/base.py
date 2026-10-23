"""Dataset contract shared by every method and the evaluator.

A split lives in splits/<name>/ and is committed to git so everyone evaluates the same images:
  references.csv     image_id,frame_index,rel_path
  queries.csv        image_id,frame_index,rel_path
  ground_truth.json  {query_id: [correct reference image_ids]}
  meta.json          how the split was built (source, protocol, subsampling, counts)
rel_path is relative to the dataset root (config `data_root`), so splits are machine-independent.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.config import REPO_ROOT

SPLITS_DIR = REPO_ROOT / "splits"
CSV_FIELDS = ("image_id", "frame_index", "rel_path")


@dataclass(frozen=True)
class ImageRecord:
    image_id: str      # globally unique, e.g. "nordland_ref_0010000"
    frame_index: int   # position along the traversal in the source dataset
    rel_path: str      # relative to data_root, forward slashes
    path: Path | None = None  # absolute path, filled in by load_split()


@dataclass
class VPRSplit:
    name: str
    references: list[ImageRecord]
    queries: list[ImageRecord]
    ground_truth: dict[str, list[str]]  # query_id -> correct reference ids
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        ref_ids = {r.image_id for r in self.references}
        q_ids = {q.image_id for q in self.queries}
        if len(ref_ids) != len(self.references) or len(q_ids) != len(self.queries):
            raise ValueError(f"split {self.name}: duplicate image_ids")
        if set(self.ground_truth) != q_ids:
            raise ValueError(f"split {self.name}: ground_truth keys must equal the query ids")
        for qid, refs in self.ground_truth.items():
            if not refs:
                raise ValueError(f"split {self.name}: query {qid} has no correct reference")
            missing = set(refs) - ref_ids
            if missing:
                raise ValueError(f"split {self.name}: query {qid} references unknown ids {sorted(missing)[:3]}")


def _write_csv(path: Path, records: list[ImageRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in records:
            w.writerow([r.image_id, r.frame_index, r.rel_path])


def _read_csv(path: Path, data_root: Path | None) -> list[ImageRecord]:
    with open(path, newline="", encoding="utf-8") as f:
        return [ImageRecord(row["image_id"], int(row["frame_index"]), row["rel_path"],
                            (data_root / row["rel_path"]) if data_root else None)
                for row in csv.DictReader(f)]


def save_split(split: VPRSplit, splits_dir: Path = SPLITS_DIR) -> Path:
    split.validate()
    out = splits_dir / split.name
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "references.csv", split.references)
    _write_csv(out / "queries.csv", split.queries)
    with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(split.ground_truth, f, indent=0)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(split.meta, f, indent=2)
    return out


def load_split(name: str, data_root: str | Path | None = None, splits_dir: Path = SPLITS_DIR) -> VPRSplit:
    d = splits_dir / name
    if not d.exists():
        available = sorted(p.name for p in splits_dir.iterdir() if p.is_dir()) if splits_dir.exists() else []
        raise FileNotFoundError(f"No split at {d}. Available: {available}")
    root = Path(data_root) if data_root else None
    with open(d / "ground_truth.json", encoding="utf-8") as f:
        gt = json.load(f)
    with open(d / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    split = VPRSplit(name, _read_csv(d / "references.csv", root), _read_csv(d / "queries.csv", root), gt, meta)
    split.validate()
    return split
