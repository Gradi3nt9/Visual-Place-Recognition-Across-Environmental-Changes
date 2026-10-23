"""Run one retrieval method on one split and save a rankings file.

  python scripts/run_retrieval.py --method random --split nordland_smoke
  python scripts/run_retrieval.py --method sift_ransac --split nordland_smoke --set methods.sift_ransac.ratio_test=0.8

Output: results/rankings/<split>__<method>[__<tag>].json  (evaluate with scripts/evaluate.py)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from src.datasets.base import load_split  # noqa: E402
from src.methods import METHODS  # noqa: E402
from src.utils.config import load_config, method_config, resolve  # noqa: E402
from src.utils.rankings import save_rankings  # noqa: E402
from src.utils.repro import run_metadata, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. ratio0.8")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries (debugging; not evaluable)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    split = load_split(args.split, resolve(cfg["data_root"]))
    mcfg = method_config(cfg, args.method)
    mcfg.setdefault("seed", cfg["seed"])
    method = METHODS[args.method](mcfg, resolve(cfg["cache_dir"]))

    t0 = time.perf_counter()
    index_stats = method.index(split.references)
    index_seconds = time.perf_counter() - t0

    queries = split.queries[: args.limit] if args.limit else split.queries
    out = {}
    for q in tqdm(queries, desc=f"{args.method} on {args.split}"):
        res = method.retrieve(q, cfg["top_k"])
        out[q.image_id] = {"ranked": [[rid, float(s)] for rid, s in res.ranked], "timings": res.timings}

    meta = {
        "method": args.method, "method_config": mcfg, "split": args.split,
        "split_counts": split.meta.get("counts"), "top_k": cfg["top_k"],
        "index_seconds": index_seconds, "index_stats": index_stats,
        "overrides": args.set, "limit": args.limit, **run_metadata(),
    }
    name = f"{args.split}__{args.method}" + (f"__{args.tag}" if args.tag else "")
    path = save_rankings(resolve(cfg["results_dir"]) / "rankings" / f"{name}.json", meta, out)
    print(f"saved {path}  (index {index_seconds:.1f}s, {len(out)} queries)")


if __name__ == "__main__":
    main()
