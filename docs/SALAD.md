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

Confirm descriptor checks pass with the actual model; keep decision F open until then.
Coordinate a shared development split with Persons 1 and 4, verify seasons with images,
and record hardware when comparing against SIFT. Smoke scores are correctness checks only.
Pin `methods.salad.hub_repo` to a reviewed upstream tag/commit using torch.hub's
`owner/repo:revision` syntax before final experiments; the scaffold default tracks upstream.

Official sources: [hub entry point](https://github.com/serizba/salad/blob/main/hubconf.py),
[preprocessing/evaluation](https://github.com/serizba/salad/blob/main/eval.py),
[descriptor normalization](https://github.com/serizba/salad/blob/main/models/aggregators/salad.py).
