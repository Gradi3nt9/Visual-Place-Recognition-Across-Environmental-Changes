"""Check a downloaded Nordland copy against the committed splits, and package it for the team.

  python scripts/check_nordland.py                                 # verify data_root
  python scripts/check_nordland.py --season-sheet                  # results/figures/nordland_seasons.png
  python scripts/check_nordland.py --zip nordland_smoke            # cache/nordland_smoke_images.zip (200 images)
  python scripts/check_nordland.py --zip nordland_full             # everything, for Drive / Colab

Verification: every image listed in splits/nordland_full exists, a sample opens as 640x360
(VPR-Bench resolution), and the ref/ and query/ folders hold no unexpected extra files.
"""
from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.datasets.base import load_split  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

EXPECTED_SIZE = (640, 360)


def verify(data_root: Path, sample: int, seed: int) -> bool:
    full = load_split("nordland_full", data_root)
    records = full.references + full.queries
    missing = [r.rel_path for r in records if not r.path.exists()]
    print(f"data_root: {data_root}")
    print(f"listed in nordland_full: {len(full.references)} refs + {len(full.queries)} queries; missing: {len(missing)}")
    for p in missing[:5]:
        print(f"  missing {p}")

    listed = {r.rel_path for r in records}
    on_disk = {f"{d}/{p.name}" for d in ("ref", "query") if (data_root / d).is_dir()
               for p in (data_root / d).iterdir() if p.is_file()}
    extra = sorted(on_disk - listed)
    print(f"files on disk: {len(on_disk)}; not in split: {len(extra)}" + (f" (e.g. {extra[0]})" if extra else ""))

    present = [r for r in records if r.path.exists()]
    bad = []
    for r in random.Random(seed).sample(present, min(sample, len(present))):
        try:
            with Image.open(r.path) as im:
                im.verify()
            with Image.open(r.path) as im:
                if im.size != EXPECTED_SIZE:
                    bad.append(f"{r.rel_path}: size {im.size}")
        except Exception as e:  # unreadable/truncated file
            bad.append(f"{r.rel_path}: {e}")
    print(f"sampled {min(sample, len(present))} images: {len(bad)} problems")
    for b in bad[:5]:
        print(f"  {b}")

    ok = not missing and not bad
    print("OK" if ok else "NOT OK")
    return ok


def season_sheet(data_root: Path, out: Path, n: int = 4) -> None:
    """Query/reference pairs side by side (top row ref/, bottom row query/) to confirm which is summer/winter."""
    full = load_split("nordland_full", data_root)
    ref_by_id = {r.image_id: r for r in full.references}
    step = len(full.queries) // (n + 1)
    # middle correct reference = the exact matching frame (10i)
    pairs = [(ref_by_id[sorted(full.ground_truth[q.image_id])[len(full.ground_truth[q.image_id]) // 2]], q)
             for q in full.queries[step::step][:n]]
    w, h = 320, 180
    sheet = Image.new("RGB", (w * n, h * 2), "white")
    for i, (ref, q) in enumerate(pairs):
        sheet.paste(Image.open(ref.path).convert("RGB").resize((w, h)), (i * w, 0))
        sheet.paste(Image.open(q.path).convert("RGB").resize((w, h)), (i * w, h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}  (top row = ref/, bottom row = query/; frames {[q.frame_index for _, q in pairs]})")


def make_zip(data_root: Path, split_name: str, out: Path) -> None:
    split = load_split(split_name, data_root)
    records = split.references + split.queries
    missing = [r.rel_path for r in records if not r.path.exists()]
    if missing:
        sys.exit(f"{len(missing)} images of {split_name} missing under {data_root} (e.g. {missing[0]}); run verify first")
    out.parent.mkdir(parents=True, exist_ok=True)
    # JPEGs are already compressed, so store them uncompressed (much faster, same size).
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for r in records:
            zf.write(r.path, r.rel_path)
    print(f"wrote {out}: {len(records)} images, {out.stat().st_size / 1e6:.1f} MB "
          f"(unzip so that ref/ and query/ sit directly inside data_root)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", type=int, default=200, help="how many images to open during verification")
    ap.add_argument("--season-sheet", action="store_true")
    ap.add_argument("--zip", metavar="SPLIT", help="package the images of this split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = resolve(cfg["data_root"])
    if args.season_sheet:
        season_sheet(data_root, resolve(cfg["results_dir"]) / "figures" / "nordland_seasons.png")
    elif args.zip:
        make_zip(data_root, args.zip, resolve(cfg["cache_dir"]) / f"{args.zip}_images.zip")
    else:
        sys.exit(0 if verify(data_root, args.sample, cfg["seed"]) else 1)


if __name__ == "__main__":
    main()
