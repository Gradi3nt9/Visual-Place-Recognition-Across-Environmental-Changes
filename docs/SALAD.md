# Person 3: SALAD baseline

The implementation is in `src/methods/salad.py`; use the shared runner and evaluator.
Follow `DECISIONS.md` and `PROJECT_HANDOFF.md`. Do not train the network or start hybrids.

## First run

Use Python 3.10 with the dependencies in `requirements.txt`. Put Nordland images under
`data/nordland/{ref,query}/`, or set `VPR_DATA_ROOT` to their parent directory.

```bash
python -m pytest -q
# First check: index 100 references and retrieve one query. Not evaluable as a full split.
python scripts/run_retrieval.py --method salad --split nordland_smoke --limit 1 --tag debug --set methods.salad.batch_size=1
# Then run the entire correctness split and evaluate.
python scripts/run_retrieval.py --method salad --split nordland_smoke
python scripts/evaluate.py results/rankings/nordland_smoke__salad.json
```

The first model load needs internet to fetch official code and weights through torch.hub.
`device=auto` selects CUDA when available, otherwise CPU; batch size is configurable.
Model setup is included in offline indexing, including on cache hits.

## Implementation

- Official pretrained `dinov2_salad`, evaluation mode, inference mode; no training.
- RGB input, bilinear resize to 322 × 322, tensor conversion, ImageNet normalization.
- Float32 inference on CPU or CUDA. Official `eval.py` uses CUDA float16 autocast;
  this baseline deliberately uses float32 and logs the precision in index statistics.
- Every output must be finite, 8448-dimensional and unit length within tolerance.
  Invalid images fail with an image ID/path rather than silently altering the split.
- Reference descriptors and ordered IDs are cached under `cache/salad/`. The key includes
  configuration, ordered references, resolved paths, file sizes and modification times.
  Moving the dataset rebuilds the cache. Modified code/weights require clearing the cache
  or changing the model repository revision/configuration.
- Queries are encoded afresh under `extract`. The CPU descriptor copy waits for CUDA
  completion. CPU matrix multiplication and stable descending sorting run under `search`.
- Cosine similarity is the dot product of unit descriptors. Squared L2 distance is
  `2 - 2 * cosine`, so both produce the same ordering. Ties preserve reference order.

Tests mock descriptor extraction to exercise cache reuse/invalidation, uncached queries,
ranking, timing stages, and invalid outputs without downloading weights. They do not
substitute for a real-image/model smoke run.

## Before reporting results

Real-model descriptor checks passed; see decision 8 in `DECISIONS.md`.
Use the shared `nordland_dev1` split for midterm results and record hardware when
comparing against SIFT. Smoke scores are correctness checks only.
Pin `methods.salad.hub_repo` to a reviewed upstream tag/commit using torch.hub's
`owner/repo:revision` syntax before final experiments; the scaffold default tracks upstream.

Official sources: [hub entry point](https://github.com/serizba/salad/blob/main/hubconf.py),
[preprocessing/evaluation](https://github.com/serizba/salad/blob/main/eval.py),
[descriptor normalization](https://github.com/serizba/salad/blob/main/models/aggregators/salad.py).


## Real-data validation — October 27, 2026

Used official pretrained DINOv2 + SALAD with 322×322 inputs and float32 inference
on a Google Colab Tesla T4. Cosine similarity search ran on CPU. Query descriptors
were never cached. The following measurements were reported from Colab:

| Split | Queries | References | R@1 (%) | R@5 (%) | R@10 (%) | Runtime/query (ms) | Indexing (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nordland_smoke | 100 | 100 | 90.00 | 100.00 | 100.00 | 47.6 | 22.6 |
| nordland_dev1 | 276 | 2,753 | 86.23 | 97.10 | 98.55 | 56.1 | 113.3 |
| nordland_full | 2,760 | 27,592 | 78.22 | 90.62 | 93.33 | 133.6 | 1179.6 |

Runtime/query includes query extraction and search; database indexing is reported
separately. These initial runs had zero reference-cache hits. Smoke results are
correctness checks, not midterm experimental results. Published Nordland numbers
must be compared against the full protocol with matching evaluation settings;
that comparison is still pending.

A repeated smoke run loaded all 100 references from cache and retained the same
recall (90% R@1, 100% R@5/R@10). Query descriptors were still computed afresh.
The repeated run measured 57.3 ms/query and 9.4 s for offline preparation, which
includes model loading and cache reading.

All real descriptors passed the finite-value, 8448-dimension, and L2 unit-norm
checks (absolute tolerance 1e-3).

Code commit: `36e0480b`. Runs were marked dirty; `git status` showed untracked
`results/`. Upstream SALAD and DINOv2 code used unpinned main revisions; exact
upstream commits were not recorded. Runtime output identified Python 3.13 in
Colab, so this was not the repository's pinned Python 3.10 environment.

Setup also required `pytorch-metric-learning`, installed separately in Colab.
Exact installed package versions were not captured in the supplied evaluation
tables. Save rankings, tables, and per-query outputs from Colab to the shared
Drive before ending the session; these artifacts are not included here.
