"""Evaluate one or more rankings files against their split's ground truth.

  python scripts/evaluate.py results/rankings/nordland_smoke__random.json [more.json ...]

Prints a Markdown table (Recall@K + runtime/query) and writes:
  results/tables/<name>.md and .csv    the table
  results/per_query/<name>.json        first-correct rank per query, for qualitative analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.base import load_split  # noqa: E402
from src.evaluation.recall import evaluate_recall  # noqa: E402
from src.evaluation.runtime import summarize_runtime  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.rankings import load_rankings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rankings", nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="table name (default: derived from the inputs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ks = cfg["recall_ks"]
    results_dir = resolve(cfg["results_dir"])
    rows = []
    for path in args.rankings:
        payload = load_rankings(path)
        meta = payload["meta"]
        if meta.get("limit"):
            sys.exit(f"{path} was produced with --limit and covers only part of the split")
        split = load_split(meta["split"])
        ev = evaluate_recall(payload["queries"], split.ground_truth, ks, {r.image_id for r in split.references})
        rt = summarize_runtime(payload["queries"])
        name = Path(path).stem
        per_q = results_dir / "per_query" / f"{name}.json"
        per_q.parent.mkdir(parents=True, exist_ok=True)
        per_q.write_text(json.dumps(ev["per_query"], indent=0), encoding="utf-8")
        row = {"run": name, "method": meta["method"], "split": meta["split"],
               "queries": ev["n_queries"], "references": len(split.references)}
        row.update({f"R@{k}": round(100 * ev["recall"][k], 2) for k in ks})
        for stage, s in rt.items():
            row[f"{stage}_ms"] = round(1000 * s["mean"], 1)
        row["index_s"] = round(meta.get("index_seconds", 0.0), 1)
        row["git_commit"] = (meta.get("git_commit") or "")[:8] + ("*" if meta.get("git_dirty") else "")
        rows.append(row)

    cols = list(dict.fromkeys(c for r in rows for c in r))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    table = "\n".join(lines)
    print(table)
    print("\nR@K in %. *_ms = mean wall-clock per query (extract/search/verify, see docs/DECISIONS.md). "
          "index_s = one-off database preparation (may include cache hits). '*' after commit = uncommitted changes.")

    out_name = args.out or (Path(args.rankings[0]).stem if len(rows) == 1 else "comparison")
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / f"{out_name}.md").write_text(table + "\n", encoding="utf-8")
    with open(tables / f"{out_name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
