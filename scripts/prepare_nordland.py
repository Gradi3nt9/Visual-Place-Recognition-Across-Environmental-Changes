"""Download the official Nordland ground truth and write a split to splits/<name>/.

Examples:
  # 100-query smoke test (database = every 10th frame, 1 correct ref per query)
  python scripts/prepare_nordland.py --name nordland_smoke --query-start 1000 --num-queries 100 --ref-stride 10
  # full official protocol (2760 queries vs 27592 references)
  python scripts/prepare_nordland.py --name nordland_full

The images themselves are NOT downloaded here (see README: Dataset).
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.base import save_split  # noqa: E402
from src.datasets.nordland import GT_FILES, GT_URL, build_split  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402


def fetch_gt(gt_dir: Path) -> None:
    gt_dir.mkdir(parents=True, exist_ok=True)
    for name in GT_FILES:
        dest = gt_dir / name
        if not dest.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(GT_URL.format(name=name), dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="split name (folder under splits/)")
    ap.add_argument("--query-start", type=int, default=0)
    ap.add_argument("--num-queries", type=int, default=None)
    ap.add_argument("--ref-stride", type=int, default=1)
    ap.add_argument("--ref-margin", type=int, default=0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gt_dir = resolve(cfg["cache_dir"]) / "nordland_gt"
    fetch_gt(gt_dir)
    split = build_split(gt_dir, args.name, args.query_start, args.num_queries, args.ref_stride, args.ref_margin)
    out = save_split(split)
    c = split.meta["counts"]
    print(f"wrote {out}: {c['queries']} queries, {c['references']} references, "
          f"{c['queries_dropped_no_gt']} queries dropped (no correct ref after subsampling)")

    data_root = resolve(cfg["data_root"])
    missing = [r.rel_path for r in split.references + split.queries if not (data_root / r.rel_path).exists()]
    if missing:
        print(f"note: {len(missing)} images not found under {data_root} (e.g. {missing[0]}); "
              "download the dataset before running image-based methods")


if __name__ == "__main__":
    main()
