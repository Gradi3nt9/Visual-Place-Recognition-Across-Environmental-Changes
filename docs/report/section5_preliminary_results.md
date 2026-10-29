# 5. Preliminary Results (draft)

**Table 1. Preliminary retrieval results on `nordland_dev1`.** 276 winter queries against 2,753 summer reference frames, with the official ±1-frame ground truth and no dropped queries. Random was run on a MacBook Pro (CPU), SIFT+RANSAC on a MacBook Air (CPU), and SALAD on a Google Colab T4 GPU, so runtimes are not directly comparable across rows. Runtimes are per-query means; database indexing is excluded. These results are from one contiguous subset of one route and are preliminary.

| Method | R@1 (%) | R@5 (%) | R@10 (%) | Extract (ms/query) | Verify (ms/query) | Total (ms/query) |
|---|---:|---:|---:|---:|---:|---:|
| Random (floor) | 0.00 | 0.72 | 1.09 | not measured | not measured | 0.2 |
| SIFT + RANSAC | 18.84 | 22.83 | 24.64 | 35.4 | 7143.0 | 7178.9 |
| SALAD | [pending] | [pending] | [pending] | [pending] | n/a | [pending] |

Notes for the report:
- The random row's total covers search only, since it does not extract features, so "not measured" is accurate for extract.
- The SALAD row has no verify step, so n/a is correct there.
- Fill the SALAD row only from its saved rankings file. Do not use the numbers from `docs/SALAD.md` unless that file never arrives; if it doesn't, label them as unreproduced.
