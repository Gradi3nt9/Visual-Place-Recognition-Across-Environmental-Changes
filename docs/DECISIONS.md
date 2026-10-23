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

## Open (needs a decision, owner in brackets)

| # | Question | Owner | Notes |
|---|---|---|---|
| A | Verify seasons: is `ref/` summer and `query/` winter? | Person 1 | Look at a few images once downloaded; update `meta.json` text in `nordland.py`. |
| B | Development split(s) for the midterm numbers | Person 1 + 4 | Full protocol is 2,760 × 27,592. Exhaustive SIFT on that is ~76M RANSAC calls, infeasible. Options: contiguous route segment at stride 1 (keeps ±1 tolerance), or larger segment at stride 10. Time SIFT on the smoke split first, then size the dev split. |
| C | Dataset location on each machine | everyone | Default `data/nordland/` (gitignored), override with `VPR_DATA_ROOT`. Must contain `ref/` and `query/`. |
| D | Compute for SALAD | team | Dev laptop is CPU-only. Is there a CUDA GPU, Colab, or a CSL machine? SALAD on the full 30k images on CPU is likely hours. |
| E | SIFT image size | Person 2 | `max_image_side` (null = native). Check Nordland's native resolution first. |
| F | Final SALAD similarity/search | Person 3 | Cosine on L2-normalized descriptors (equivalent to SALAD's faiss L2). Confirm descriptors are normalized in practice. |
