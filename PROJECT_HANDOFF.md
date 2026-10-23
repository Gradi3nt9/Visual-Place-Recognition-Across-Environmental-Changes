# CS 566 Computer Vision Final Project — Handoff Document

## 1. Project Context

This is a four-person undergraduate Computer Vision course final project at UW–Madison.

**Project topic:** Visual Place Recognition (VPR) Across Environmental Changes.

**Current date:** October 23, 2026.

**Immediate deadline:** Midterm progress report due **October 29, 2026**.

We are currently behind schedule and have approximately six days to produce the first working implementation and preliminary results. The immediate priority is therefore **a simple, correct, reproducible baseline implementation**, not building the entire final project.

The submitted proposal states that by the October 29 midterm we aim to have our **primary baselines implemented with preliminary results**.

The standard hybrid approach and proposed confidence-adaptive method are scheduled for **after the midterm**.

---

# 2. Original Research Problem

Visual place recognition is challenging because the appearance of the same location can change significantly across environmental conditions and viewpoints, such as different seasons, times of day, and camera perspectives.

The project investigates how different visual representations address these variations, particularly whether learned image representations provide more robust retrieval than classical local-feature methods.

We also want to investigate whether geometric verification can complement learned image retrieval by filtering false matches and confirming spatial consistency.

Our central hypothesis is:

> Learned representations will improve robustness to environmental and viewpoint changes, while local geometric verification may provide complementary spatial evidence when reliable local correspondences are available.

The broader research question is not simply "which method gets the highest accuracy?"

We ultimately want to understand:

> **When does geometric verification help a strong learned VPR representation, and when does it provide unreliable information?**

---

# 3. Submitted Project Plan

The submitted proposal defines four methods.

## Method A — Classical baseline

SIFT local features + descriptor matching + RANSAC geometric verification.

Conceptually:

Query image
→ SIFT
→ local descriptors
→ descriptor matching against database image
→ geometric verification using RANSAC
→ geometric consistency score
→ ranked database images

---

## Method B — Learned global baseline

Use a pretrained modern VPR model such as SALAD or BoQ to produce one global descriptor for each image.

Conceptually:

Image
→ pretrained VPR network
→ global descriptor

For a query:

Query descriptor
→ compare against database descriptors
→ similarity scores
→ ranked database images

---

## Method C — Standard hybrid

Use the learned representation for initial retrieval.

Retrieve top-K database candidates.

Run local feature matching + RANSAC only on those candidates.

Use geometric consistency to rerank the top-K candidates.

Conceptually:

Learned retrieval
→ top-K
→ SIFT matching
→ RANSAC
→ geometric reranking

K is an experimental variable rather than something that must be fixed permanently.

---

## Method D — Confidence-adaptive hybrid

This is our proposed experimental extension.

The hypothesis is that geometric verification should not necessarily have equal influence for every query/candidate pair.

If geometric evidence is strong, geometry should influence the final ranking more.

If geometric evidence is weak, the system should rely more heavily on the learned global similarity.

Possible confidence signals include:

- RANSAC inlier count
- RANSAC inlier ratio
- descriptor-match quality
- potentially spatial distribution of inliers

A conceptual scoring rule might eventually look like:

final_score =
(1 - w) * global_score
+ w * geometric_score

where `w` depends on geometric confidence.

**IMPORTANT:** This formula is conceptual only.

The exact confidence function and score normalization have NOT been decided yet.

Do not implement this method before the midterm baselines unless explicitly instructed.

---

# 4. Scope of the October 29 Midterm

The immediate objective is NOT to implement all four methods.

For the midterm, we need:

1. Dataset preparation
2. Classical SIFT/RANSAC baseline
3. Learned VPR baseline
4. Common evaluation framework
5. Preliminary results

Ideally the midterm should contain something resembling:

| Method | Recall@1 | Recall@5 | Runtime/query |
|---|---:|---:|---:|
| SIFT + RANSAC | result | result | result |
| SALAD | result | result | result |

We also want several qualitative retrieval examples if time permits.

The following are explicitly **post-midterm work**:

- standard hybrid reranking
- adaptive confidence weighting
- K ablations
- fixed-weight ablations
- extensive failure analysis
- full Oxford RobotCar experiments

Do NOT allow those tasks to distract from getting the two baselines working.

---

# 5. Current Technical Decisions

Unless evidence from the actual implementation reveals a serious problem, use the following choices.

## Dataset for midterm

**Nordland**

Use:

- **Summer traversal as the reference/database**
- **Winter traversal as the query set**

This provides a strong seasonal appearance change while preserving the same train route.

We may eventually evaluate additional season combinations.

The final project also intends to use Oxford RobotCar, but Oxford is not required for the immediate midterm if doing so jeopardizes getting working preliminary results.

---

# 6. Nordland Evaluation Warning

Do NOT invent a universal GPS threshold such as "within 25 meters."

Nordland consists of aligned traversals of the same train route across seasons.

The evaluation protocol must reflect the dataset's actual frame/route correspondence.

Before final evaluation:

1. inspect the Nordland representation being used;
2. inspect established VPR evaluation conventions or the evaluation implementation associated with the selected model;
3. determine how reference/query frames correspond;
4. explicitly document the ground-truth tolerance used.

If the dataset is subsampled, make sure the ground-truth correspondence is transformed consistently.

The exact subsampling interval and frame tolerance have deliberately NOT yet been fixed.

Determine them based on the actual dataset representation and computational feasibility rather than choosing arbitrary values.

---

# 7. Learned Baseline Decision

Use **SALAD** as the learned VPR baseline unless there is a major implementation blocker.

SALAD = Sinkhorn Algorithm for Locally Aggregated Descriptors.

It uses a DINOv2-based feature extractor and aggregates local features into a global VPR descriptor.

Use the official pretrained implementation/weights rather than training SALAD from scratch.

The purpose of this project is NOT to retrain a modern VPR foundation model.

For every database image:

image
→ SALAD
→ descriptor

Store/cache the descriptors.

For every query:

query
→ SALAD
→ descriptor
→ similarity against cached database descriptors
→ ranked database IDs

Use the similarity convention expected by the official SALAD implementation. If descriptors are normalized and cosine similarity is appropriate, use cosine similarity.

Do not silently change the model architecture.

---

# 8. Classical Baseline

Use **OpenCV SIFT**.

Pipeline:

image
→ SIFT keypoints/descriptors
→ descriptor matching
→ ambiguous-match filtering
→ geometric verification
→ geometric score

Initial matcher:

**k-nearest-neighbor descriptor matching with k=2**

Initial ambiguous-match filter:

**Lowe ratio test**

Initial ratio threshold:

**0.75**

This threshold can later be tuned or ablated if necessary. For the immediate baseline, prioritize a standard reproducible implementation.

---

# 9. Geometric Verification

Initial geometric model:

**Fundamental matrix estimated with RANSAC.**

Reasoning:

Nordland contains general outdoor 3D scenes and camera translation. A single homography assumes a more restrictive planar/projective relationship.

Therefore the initial baseline should use a fundamental matrix rather than assuming that the scene is planar.

Pipeline:

SIFT matches
→ Lowe-filtered correspondences
→ fundamental matrix estimation with RANSAC
→ inlier mask
→ number of geometrically consistent matches

The initial classical retrieval score should be:

**RANSAC inlier count**

Higher inlier count = stronger candidate.

However, keep the code modular enough that the scoring function can later be changed to something such as:

- inlier ratio
- normalized inlier count
- another geometric-confidence measure

Do NOT overengineer this before the first results.

Handle failure cases safely:

- missing descriptors
- too few matches
- failed fundamental-matrix estimation
- malformed images

These should produce an appropriate low/zero geometric score rather than crashing the entire experiment.

---

# 10. Computational Considerations

Exhaustive SIFT matching can become expensive because every query could potentially be compared with every database image.

Therefore:

**Cache all extracted SIFT features.**

Do NOT repeatedly run SIFT on the same database image.

Likewise:

**Cache all SALAD database descriptors.**

For initial development, use a small subset of Nordland.

The workflow should be:

small smoke-test subset
→ verify correctness
→ moderate development subset
→ measure runtime
→ scale as far as practical

Do not immediately launch a huge exhaustive experiment without estimating its cost.

If Nordland must be subsampled for the midterm, that is acceptable as long as:

1. it is documented;
2. both methods use the same evaluation set;
3. ground truth remains correct;
4. the report clearly identifies the results as preliminary.

---

# 11. Required Common Interface

All retrieval methods should eventually conform to the same conceptual interface.

Given:

- query image/query ID
- database

Return:

- database IDs ranked from most likely match to least likely match
- optionally associated scores

Conceptually:

```python
results = retrieve(query, database)

# Example
[
    ("summer_001253", 0.94),
    ("summer_001250", 0.91),
    ("summer_001811", 0.87),
]
```

The evaluator should NOT need to know whether these rankings came from:

- SIFT
- SALAD
- hybrid
- adaptive hybrid

Keep retrieval and evaluation separated.

---

# 12. Evaluation Metrics

Primary metrics from the submitted proposal:

## Recall@1

Fraction of queries for which a correct reference image appears at rank 1.

## Recall@5

Fraction of queries for which at least one correct reference image appears within the first five retrieved candidates.

## Runtime per query

Measure retrieval runtime so we can eventually study the computational cost introduced by geometric verification/reranking.

Be precise about what runtime includes.

For example, distinguish if useful between:

- feature extraction
- retrieval/search
- geometric verification

Do not compare cached and uncached pipelines unfairly without explaining what is measured.

---

# 13. Repository Architecture

Prefer a clean modular structure approximately like:

```text
vpr-project/
│
├── README.md
├── PROJECT_HANDOFF.md
├── requirements.txt / pyproject.toml
│
├── configs/
│   └── ...
│
├── data/
│   ├── nordland/
│   └── ...
│
├── src/
│   ├── datasets/
│   │   └── nordland.py
│   │
│   ├── methods/
│   │   ├── sift_ransac.py
│   │   └── salad.py
│   │
│   ├── evaluation/
│   │   ├── recall.py
│   │   └── runtime.py
│   │
│   └── utils/
│
├── scripts/
│   ├── prepare_nordland.py
│   ├── run_sift_baseline.py
│   ├── run_salad_baseline.py
│   └── evaluate.py
│
├── cache/
│   ├── sift/
│   └── salad/
│
└── results/
    ├── rankings/
    ├── tables/
    └── figures/
```

This is a recommendation, not an immutable requirement.

If the existing repository already has a sensible structure, integrate with it rather than unnecessarily rewriting everything.

---

# 14. Four-Person Work Division

There are four students.

Parallelization should roughly follow:

## Person 1 — Dataset + Ground Truth

Responsibilities:

- acquire/prepare Nordland;
- summer reference traversal;
- winter query traversal;
- investigate established correspondence/evaluation protocol;
- implement dataset loader;
- define database/query IDs;
- implement ground-truth mapping/tolerance;
- make sure subsampling preserves correspondence.

Deliverable:

A reproducible dataset/evaluation representation that the other methods can consume.

---

## Person 2 — Classical Baseline

Responsibilities:

- OpenCV SIFT;
- feature extraction;
- feature caching;
- kNN matching;
- Lowe ratio test;
- fundamental matrix + RANSAC;
- inlier scoring;
- database ranking.

Deliverable:

```text
query
→ ranked database IDs + geometric scores
```

---

## Person 3 — SALAD Baseline

Responsibilities:

- integrate pretrained SALAD;
- reproduce inference correctly;
- database descriptor extraction;
- descriptor caching;
- query descriptor extraction;
- similarity search;
- database ranking.

Deliverable:

```text
query
→ ranked database IDs + learned similarity scores
```

---

## Person 4 — Integration + Evaluation

Responsibilities:

- common interfaces;
- Recall@1;
- Recall@5;
- runtime measurement;
- integration;
- result serialization;
- retrieval visualization;
- preliminary tables;
- help unblock other components.

Deliverable:

Both baselines should flow through the same evaluator.

---

# 15. Immediate Schedule

## October 23

Goal:

- repository/environment working;
- Nordland accessible or download/preparation underway;
- SIFT works on two example images;
- SALAD generates a descriptor for one image;
- common evaluator/interface established.

Do NOT aim for full-scale results yet.

---

## October 24

Goal:

Get both pipelines working end-to-end on a **small subset**.

Desired output:

```text
SIFT/RANSAC
Recall@1: ...
Recall@5: ...

SALAD
Recall@1: ...
Recall@5: ...
```

These numbers may be preliminary and based on a small development subset.

---

## October 25–26

Goal:

- fix correctness problems;
- cache features/descriptors;
- increase evaluation size;
- record runtime;
- inspect retrievals manually;
- produce qualitative examples;
- make experiments reproducible.

---

## October 27

**Experiment freeze for midterm.**

Produce the preliminary results that will go into the report.

Do not add unnecessary new features after this point unless required to fix a correctness problem.

---

## October 28

Write/finalize midterm report.

Report should describe **what was actually implemented**, not merely repeat the proposal.

For example:

Bad:

> We will select SALAD or BoQ.

Good:

> We selected SALAD as our learned VPR baseline and used pretrained weights...

Bad:

> We will use SIFT and RANSAC.

Good:

> We implemented SIFT feature extraction using OpenCV...

Include actual:

- dataset configuration;
- number of query/reference images;
- subsampling;
- ground-truth protocol;
- SIFT parameters;
- RANSAC parameters;
- SALAD configuration;
- Recall@1;
- Recall@5;
- runtime;
- preliminary observations;
- implementation difficulties/changes.

---

## October 29

Final validation and submission.

---

# 16. Expected Midterm Report Structure

The earlier report skeleton contained the entire final-project plan. For the actual midterm, reorganize it around completed work.

Recommended structure:

## 1. Summary of Progress

What was actually completed by October 29.

Mention significant deviations from the original proposal.

Summarize preliminary findings.

## 2. Dataset and Evaluation Setup

Describe:

- Nordland;
- summer/winter configuration;
- number of images;
- sampling;
- database/query setup;
- ground truth;
- evaluation protocol.

## 3. Classical SIFT + RANSAC Baseline

Describe actual implementation.

Include actual:

- SIFT implementation;
- matching;
- ratio threshold;
- RANSAC;
- geometric model;
- ranking score.

## 4. Learned SALAD Baseline

Describe:

- pretrained model;
- descriptor generation;
- database descriptor caching;
- similarity search.

## 5. Preliminary Results

At minimum:

| Method | Recall@1 | Recall@5 | Runtime/query |
|---|---:|---:|---:|
| SIFT + RANSAC | ... | ... | ... |
| SALAD | ... | ... | ... |

Add qualitative examples if useful.

## 6. Discussion

Discuss:

- observed difference between classical and learned representations;
- failure cases;
- computational differences;
- whether results support the original hypothesis so far.

Do NOT overclaim based on preliminary experiments.

## 7. Remaining Work

After the midterm:

1. standard learned + geometric hybrid;
2. top-K reranking experiments;
3. confidence-adaptive hybrid;
4. fixed-weight comparison;
5. K ablation;
6. expanded environmental evaluation;
7. Oxford RobotCar if feasible;
8. failure-case analysis.

## 8. Updated Timeline

First half of November:

- standard hybrid;
- adaptive hybrid;
- initial comparison.

Second half of November:

- final experiments;
- ablations;
- failure analysis;
- final presentation;
- project webpage.

---

# 17. Final-Project Evaluation Plan

The submitted proposal ultimately calls for:

### Datasets

- Nordland — seasonal change
- Oxford RobotCar — illumination/weather/urban appearance

### Metrics

- Recall@1
- Recall@5
- runtime/query

### Methods

1. SIFT + RANSAC
2. SALAD learned global retrieval
3. standard hybrid
4. confidence-adaptive hybrid

### Breakdowns

Eventually investigate:

- seasonal change
- day vs. night
- weather/environmental appearance
- viewpoint change where supported by the evaluation setup

### Ablations

Eventually compare:

- adaptive weighting
- fixed global/geometric weights
- different K values

### Qualitative analysis

Find examples where:

- learned retrieval fails but geometry corrects it;
- learned retrieval is correct but geometry damages the ranking;
- both fail;
- both succeed.

These examples are important because the research question concerns **when geometry is useful**, not merely aggregate accuracy.

---

# 18. Important Scientific Constraints

## Do not claim the adaptive method is novel without evidence.

Confidence/uncertainty and adaptive geometric verification have been explored in VPR literature.

Frame this project as:

> an experimental extension/investigation of hybrid VPR techniques

rather than claiming a new state-of-the-art algorithm.

---

## Do not assume geometric verification always improves retrieval.

That is precisely what the project is investigating.

Geometry may become unreliable under:

- severe seasonal change;
- lighting changes;
- vegetation changes;
- weak local correspondences;
- large viewpoint differences.

A result where SALAD alone outperforms geometric reranking can still be scientifically interesting.

---

## Do not optimize only for the desired hypothesis.

Report negative results.

Possible meaningful outcomes include:

- adaptive > standard hybrid;
- standard hybrid > adaptive;
- SALAD > both hybrids;
- geometry helps only under particular conditions.

---

## Do not confuse global and geometric score scales.

SALAD similarity and RANSAC inlier count are not directly comparable.

Before implementing weighted combinations, scores will likely require normalization/calibration.

This is a post-midterm design problem.

---

## Do not claim adaptive weighting saves compute.

Our proposed adaptive method obtains its confidence from geometric verification.

Therefore geometry must already have been run to compute the confidence.

Unless a future method predicts whether geometry should run **before** running it, adaptive weighting itself does not avoid the geometric-computation cost.

Runtime should still be measured because hybrid reranking introduces additional cost.

---

# 19. Coding-Agent Instructions

You are being used as an implementation agent, but you should also act as a careful computer-vision engineer.

Before making major architectural changes:

1. inspect the existing repository;
2. inspect existing code;
3. understand the current environment;
4. explain the proposed change;
5. avoid rewriting working components unnecessarily.

When implementing:

- keep components modular;
- avoid duplicated feature extraction;
- cache expensive computations;
- make experiments reproducible;
- use deterministic seeds where applicable;
- expose important parameters through config/CLI rather than scattering magic constants;
- add concise comments;
- handle failure cases;
- log experimental configuration;
- save rankings/results to disk;
- preserve enough metadata to reproduce every reported number.

Do not silently modify the experimental protocol merely to make results look better.

When uncertain about a dataset convention or model implementation, inspect the authoritative source/repository rather than guessing.

---

# 20. Priority Rule

Whenever there is a conflict between:

**building more features**

and

**getting a correct reproducible baseline result before October 29**

choose:

> **the correct reproducible baseline.**

The immediate success condition is extremely simple:

> We can run the same Nordland evaluation through SIFT/RANSAC and SALAD and obtain defensible Recall@1, Recall@5, and runtime measurements.

Everything else comes afterward.

---

# 21. First Action for Claude Code

Before writing code:

1. Read this entire document.
2. Inspect the current repository recursively.
3. Read the README, dependency files, configs, and existing source code.
4. Check git status and current branch.
5. Determine what portions of the above plan already exist.
6. Do **not** delete or rewrite existing working code.
7. Produce a short status report containing:
   - current repository structure;
   - what is already implemented;
   - missing components for the Oct. 29 milestone;
   - dependency/environment issues;
   - recommended immediate implementation order.

Then begin with the **smallest end-to-end smoke test** needed to establish:

> Nordland sample → retrieval method → ranked IDs → ground-truth evaluator → Recall@K.

Do not begin the standard hybrid or adaptive method until the two baseline pipelines have produced valid preliminary results.