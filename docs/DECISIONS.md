# Decisions log

What has been decided, what is still open, and who owns it. Update this file in the PR that makes a
decision, so the midterm report can cite actual choices instead of the proposal's "we will".

## Decided (in the scaffold, change by PR + team agreement)

### 1. Image IDs and split format
- IDs: `nordland_ref_<frame:07d>` and `nordland_query_<frame:07d>`, using the frame number from the
  official file name. Dataset-prefixed so Oxford RobotCar IDs can never collide later.
- A split = `splits/<name>/{references.csv, queries.csv, ground_truth.json, meta.json}`, committed
  to git. Paths are relative to `data_root`, so the same split works on every machine.
- Code: `src/datasets/base.py` (`ImageRecord`, `VPRSplit`, `load_split`, `save_split`).

### 2. Ground truth: Nordland protocol
- We use the **VPR-Bench Nordland** release, the one SALAD's official evaluation uses.
- Ground truth comes from SALAD's official files (`datasets/Nordland/*.npy`, pinned to commit
  `6aede13`), downloaded by `scripts/prepare_nordland.py`:
  - 27,592 reference images, 2,760 query images (queries are every 10th frame);
  - query *i* is correct for reference frames **{10i-1, 10i, 10i+1}**, a tolerance of ±1 reference frame.
- `splits/nordland_full` reproduces the official ground truth exactly (checked programmatically).
- Subsets choose references first, then intersect each query's correct set with the kept references.
  Queries with no remaining correct reference are **dropped and counted** (`meta.json`), never mislabeled.

### 3. Rankings file (the method → evaluator contract)
- Every method writes `results/rankings/<split>__<method>[__tag].json` via `scripts/run_retrieval.py`.
- Per query: `ranked` = `[[ref_id, score], ...]`, best first, **higher score = better**, top `top_k`
  (default 100). Format and validation: `src/utils/rankings.py`.
- The evaluator (`scripts/evaluate.py`) reads only this file + the split. It never imports a method.
- Recall@K: correct if **any** correct reference is in the top K (same as SALAD's `get_validation_recalls`).

### 4. Runtime convention
- Per query, three wall-clock stages (`src/utils/timing.py`):
  - `extract`: computing the **query's** features. Query features are never cached.
  - `search`: comparing against the database / producing the ranking.
  - `verify`: geometric verification (SIFT+RANSAC now, hybrid reranking later).
- Database preparation (`index`) is timed once per run and reported separately as `index_s`,
  together with how many items came from cache. It is **not** part of runtime/query.
- When reporting, always state the hardware (logged in rankings `meta`) and compare SIFT vs SALAD
  on the same machine, or say clearly that they ran on different machines.

### 5. Smoke-test split
- `splits/nordland_smoke`: queries 1000-1099 vs reference frames 10000-10990, stride 10.
  100 queries × 100 references, exactly one correct reference per query.
- Purpose: correctness checks only. It is easier and smaller than the real protocol, so don't report it as a result.
- Random baseline on it: R@1 = 1%, as expected (1/100).

### 6. Image source and seasons (resolved items A and C, 2026-10-26)
- The VPR-Bench surfdrive download is offline. Images come from the
  [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) mirror
  (Patch-NetVLAD Nordland test split: 27,592 database + 27,592 query frames, aligned frame-for-frame).
- Mapping to VPR-Bench: `ref/j` = mirror database frame *j*; `query/i` = mirror query frame *10i*
  (`scripts/import_nordland_mirror.py`). Supported by the counts (27,592 and 2,760 = every 10th) and
  by `results/figures/nordland_seasons.png`: each query shows the same place as its ground-truth reference.
  Still to confirm: SALAD on `nordland_full` should land near the paper's Nordland R@1.
- **Seasons verified visually: `ref/` = summer (database), `query/` = winter (queries).**
- `scripts/check_nordland.py` on the converted copy: 30,352/30,352 images present, all open, all 640×360.
- Data location: `data/nordland/` (gitignored) or `VPR_DATA_ROOT`. Teammates use the shared zip.

### 7. Midterm evaluation split: `nordland_dev1` (item B; Person 4 please confirm in the PR)
- Official queries 1500–1775 (**276 winter queries**) vs reference frames 14,999–17,751
  (**2,753 summer references**), stride 1. Built with
  `prepare_nordland.py --name nordland_dev1 --query-start 1500 --num-queries 276`.
- Keeps the official protocol unchanged inside the segment: ±1 frame tolerance, 3 correct refs per query.
- Sized from SIFT timing on real Nordland images (laptop CPU, single process, 640×360):
  SIFT extraction 51 ms/image (~570 keypoints); kNN + ratio 0.75 + F-RANSAC **7.2 ms per pair**.
  dev1 = 276 × 2,753 pairs ≈ **1.5 h** exhaustive. The full protocol would be ≈ 153 h, so it's infeasible.
- No overlap with `nordland_smoke`, so use smoke for debugging and tuning and dev1 for reported numbers.
- Limitation for the report: one contiguous stretch of the route (10% of queries) and a smaller database
  than the full protocol, so absolute recalls aren't comparable to published Nordland numbers.
  SALAD (cheap) can also be run on `nordland_full` for that comparison.
- Early observation (10 queries × 40 refs; 10 correct, 390 wrong pairs): median RANSAC inliers
  30 for correct vs 10 for wrong pairs. SIFT does separate them across seasons, but not by much.

## Open (needs a decision, owner in brackets)

| # | Question | Owner | Notes |
|---|---|---|---|
| D | Compute for SALAD | team | Decided: Google Colab GPU. Person 1 shares the images on Drive. |
| E | SIFT image size | Person 2 | `max_image_side` (null = native). Check Nordland's native resolution first. |
| F | Final SALAD similarity/search | Person 3 | Cosine on L2-normalized descriptors (equivalent to SALAD's faiss L2). Confirm descriptors are normalized in practice. |
