"""Qualitative retrieval figures: query, each method's top-1 retrieval, and a correct reference.

  python scripts/visualize_retrievals.py --split nordland_smoke \
      results/rankings/nordland_smoke__sift_ransac.json results/rankings/nordland_smoke__salad.json

One rankings file: queries are shown as correct / wrong at top-1.
Two rankings files: queries are grouped by which method got them right at top-1
  (both correct, only the first, only the second, both wrong).
Queries are sampled with a fixed seed, so the same command gives the same figures.

Output: results/figures/<split>__<group>.png   (one figure per group)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no display needed
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.datasets.base import load_split  # noqa: E402
from src.evaluation.recall import first_correct_rank  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.rankings import load_rankings  # noqa: E402

GREEN, RED = "#2ca02c", "#d62728"


def _top1_correct(payload: dict, gt: dict[str, list[str]]) -> dict[str, bool]:
    """query id -> whether the top-1 retrieval is a correct reference."""
    return {qid: first_correct_rank([r for r, _ in e["ranked"]], set(gt[qid])) == 1
            for qid, e in payload["queries"].items()}


def _group_queries(runs: list[tuple[str, dict[str, bool]]], qids: list[str]) -> dict[str, list[str]]:
    """Split query ids into groups by which methods are correct at top-1."""
    if len(runs) == 1:
        name, ok = runs[0]
        return {f"{name}_correct": [q for q in qids if ok[q]],
                f"{name}_wrong": [q for q in qids if not ok[q]]}
    (na, oa), (nb, ob) = runs
    groups = {"both_correct": [], f"only_{na}": [], f"only_{nb}": [], "both_wrong": []}
    for q in qids:
        key = {(True, True): "both_correct", (True, False): f"only_{na}",
               (False, True): f"only_{nb}", (False, False): "both_wrong"}[(oa[q], ob[q])]
        groups[key].append(q)
    return groups


def _draw_row(axes, qid, runs_payload, gt, ref_by_id, q_by_id):
    """One query: query image, then each method's top-1 (green = correct, red = wrong)."""
    q_img = mpimg.imread(q_by_id[qid].path)
    axes[0].imshow(q_img)
    axes[0].set_title(f"query\n{qid}", fontsize=7)
    correct_ids = set(gt[qid])
    for ax, (name, payload) in zip(axes[1:], runs_payload):
        top_id, score = payload["queries"][qid]["ranked"][0]
        ok = top_id in correct_ids
        ax.imshow(mpimg.imread(ref_by_id[top_id].path))
        ax.set_title(f"{name} top-1\n{top_id}  score {score:.1f}", fontsize=7,
                     color=GREEN if ok else RED)
        for s in ax.spines.values():
            s.set_edgecolor(GREEN if ok else RED)
            s.set_linewidth(3)
    # the first correct reference, for comparison
    gt_ref = min(correct_ids, key=lambda r: ref_by_id[r].frame_index)
    axes[-1].imshow(mpimg.imread(ref_by_id[gt_ref].path))
    axes[-1].set_title(f"ground truth\n{gt_ref}", fontsize=7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rankings", nargs="+", help="one or two rankings files for the same split")
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-group", type=int, default=3, help="queries sampled per group")
    ap.add_argument("--seed", type=int, default=None, help="default: config seed")
    args = ap.parse_args()
    if len(args.rankings) > 2:
        sys.exit("give one or two rankings files")

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    split = load_split(args.split, resolve(cfg["data_root"]))
    gt = split.ground_truth
    ref_by_id = {r.image_id: r for r in split.references}
    q_by_id = {q.image_id: q for q in split.queries}

    payloads = []
    for path in args.rankings:
        p = load_rankings(path)
        if p["meta"]["split"] != args.split:
            sys.exit(f"{path} was produced on split {p['meta']['split']!r}, not {args.split!r}")
        if p["meta"].get("limit"):
            sys.exit(f"{path} was produced with --limit and covers only part of the split")
        payloads.append((p["meta"]["method"], p))
    if len(payloads) == 2 and set(payloads[0][1]["queries"]) != set(payloads[1][1]["queries"]):
        sys.exit("the two rankings files cover different query sets")

    qids = sorted(payloads[0][1]["queries"])
    runs = [(name, _top1_correct(p, gt)) for name, p in payloads]
    groups = _group_queries(runs, qids)

    out_dir = resolve(cfg["results_dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for group, members in groups.items():
        if not members:
            print(f"{group}: no queries")
            continue
        chosen = sorted(rng.sample(members, min(args.per_group, len(members))))
        n_cols = len(payloads) + 2  # query, one column per method, ground truth
        fig, axes = plt.subplots(len(chosen), n_cols, figsize=(2.6 * n_cols, 2.8 * len(chosen)), squeeze=False)
        for row_axes, qid in zip(axes, chosen):
            _draw_row(row_axes, qid, payloads, gt, ref_by_id, q_by_id)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{args.split}: {group} ({len(members)} queries, showing {len(chosen)})", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"{args.split}__{group}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"{group}: {len(members)} queries -> {path}")


if __name__ == "__main__":
    main()
