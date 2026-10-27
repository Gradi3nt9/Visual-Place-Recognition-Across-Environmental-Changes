"""Convert the gmberton/VPR-datasets-downloader Nordland mirror into the VPR-Bench layout our splits use.

The VPR-Bench download (surfdrive) is offline, so we take the images from the mirror:
  rsync://vandaldata.polito.it/sf_xl/VPR-datasets-downloader/nordland
It holds the Patch-NetVLAD Nordland test split: 27,592 database and 27,592 query frames, aligned
frame-for-frame. File names look like  images/test/database/@0@00002.3@@@@@1@@@@@@@@.jpg ,
where the 8th '@'-field is the frame index.

Mapping to VPR-Bench (the layout SALAD's ground truth refers to):
  ref/{j:07d}.jpg    <- database frame j        (j = 0 .. 27591)
  query/{i:07d}.jpg  <- queries  frame 10*i     (i = 0 .. 2759)
This follows from the counts (27,592 refs; 2,760 = every 10th query) and from the official ground
truth pairing query i with refs 10i-1..10i+1. Confirm it by checking that SALAD on nordland_full
lands near the paper's Nordland Recall@1 (see docs/DECISIONS.md).

  python scripts/import_nordland_mirror.py data/nordland_mirror      # writes into data_root
Files are hard-linked (no extra disk space) and copied only if linking fails.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve  # noqa: E402

N_FRAMES = 27592
QUERY_STRIDE = 10


def frame_index(path: Path) -> int:
    return int(path.stem.split("@")[7])


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mirror", help="folder the mirror was rsynced into (contains images/test/...)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    mirror = Path(args.mirror) / "images" / "test"
    data_root = resolve(load_config(args.config)["data_root"])
    db = {frame_index(p): p for p in (mirror / "database").glob("*.jpg")}
    qs = {frame_index(p): p for p in (mirror / "queries").glob("*.jpg")}

    missing_db = [j for j in range(N_FRAMES) if j not in db]
    missing_q = [f for f in range(0, N_FRAMES, QUERY_STRIDE) if f not in qs]
    if missing_db or missing_q:
        sys.exit(f"mirror incomplete: {len(missing_db)} database and {len(missing_q)} query frames missing "
                 f"(e.g. db {missing_db[:3]}, query {missing_q[:3]}); re-run the rsync")

    (data_root / "ref").mkdir(parents=True, exist_ok=True)
    (data_root / "query").mkdir(parents=True, exist_ok=True)
    for j in range(N_FRAMES):
        link_or_copy(db[j], data_root / "ref" / f"{j:07d}.jpg")
    for i, f in enumerate(range(0, N_FRAMES, QUERY_STRIDE)):
        link_or_copy(qs[f], data_root / "query" / f"{i:07d}.jpg")
    print(f"wrote {N_FRAMES} refs and {len(range(0, N_FRAMES, QUERY_STRIDE))} queries into {data_root}")


if __name__ == "__main__":
    main()
