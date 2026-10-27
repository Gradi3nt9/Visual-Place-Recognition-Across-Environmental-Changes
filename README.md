# Visual Place Recognition Across Environmental Changes

CS 566 (UW–Madison) course project. We compare classical local features (SIFT + RANSAC) with a
learned global descriptor (SALAD) for place recognition across seasons on Nordland. After the
midterm, we test whether geometric verification helps learned retrieval (standard and
confidence-adaptive hybrid reranking).

- Full plan and constraints: [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)
- What's decided vs. open: [docs/DECISIONS.md](docs/DECISIONS.md)

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt      # on a CUDA machine install torch/torchvision from pytorch.org first
python -m pytest -q                  # should pass with no data downloaded
```

## Dataset

We evaluate with the VPR-Bench Nordland protocol (the one SALAD's official evaluation uses). The
VPR-Bench download (surfdrive) is offline, so we take the images from the
[VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) mirror and convert
them (~1.5 GB, only the 30,352 images we need). Needs `rsync` (on Windows: run it inside WSL):

```bash
python scripts/prepare_nordland.py --name nordland_full         # downloads official ground truth
# list of the files we need (all database frames + every 10th query frame)
rsync -q rsync://vandaldata.polito.it/sf_xl/VPR-datasets-downloader/nordland/all_images_paths.txt .
grep -E 'database/|@[0-9]*0@@@@@@@@\.jpg$' all_images_paths.txt > needed.txt
rsync -rz --files-from=needed.txt rsync://vandaldata.polito.it/sf_xl/VPR-datasets-downloader/nordland/ data/nordland_mirror/
python scripts/import_nordland_mirror.py data/nordland_mirror   # -> data/nordland/ref, data/nordland/query
python scripts/check_nordland.py                                # verify everything is there
```

Teammates can skip this and use the shared Drive zip instead. Either way, this layout must exist
(or set `VPR_DATA_ROOT` to the folder holding `ref/` and `query/`):

```
data/nordland/ref/0000000.jpg ... 0027591.jpg     # 27,592 reference images
data/nordland/query/0000000.jpg ... 0002759.jpg   # 2,760 query images
```

Ground truth is not in that download. `scripts/prepare_nordland.py` fetches the official files and
writes splits. Committed splits are in `splits/` (`nordland_smoke`, `nordland_full`).

## Running

```bash
# (optional) build a new split, e.g. a dev subset
python scripts/prepare_nordland.py --name nordland_dev --query-start 1000 --num-queries 200 --ref-stride 10

# run a method -> results/rankings/<split>__<method>.json
python scripts/run_retrieval.py --method random --split nordland_smoke
python scripts/run_retrieval.py --method sift_ransac --split nordland_smoke --set methods.sift_ransac.ratio_test=0.8 --tag r08

# evaluate one or several runs -> table printed + results/tables/
python scripts/evaluate.py results/rankings/nordland_smoke__random.json results/rankings/nordland_smoke__salad.json
```

Parameters live in `configs/default.yaml`. Override them with `--set key=value`, not by editing code.
Every rankings file records its config, git commit, and machine.

## Repository layout

```
configs/default.yaml         all experiment parameters
splits/<name>/               committed dataset splits (image lists + ground truth)
src/datasets/                split format (base.py), Nordland builder (nordland.py)
src/methods/                 RetrievalMethod interface (base.py), random, sift_ransac, salad
src/evaluation/              Recall@K, runtime summary
src/utils/                   config, timing stages, rankings file I/O, run metadata
scripts/                     prepare_nordland.py, run_retrieval.py, evaluate.py
tests/                       synthetic tests for the evaluator and splits
data/  cache/  results/      local only (gitignored, except results/tables if we choose)
```

## Adding a method

Subclass `RetrievalMethod` (`src/methods/base.py`), implement `index()` and `retrieve()`, time the
query stages with `StageTimer`, register it in `src/methods/__init__.py`, and add a config section.
The runner and evaluator then work unchanged.

## Team workflow

| Person | Role | Branch |
|---|---|---|
| 1 | Dataset + ground truth (`src/datasets`, `scripts/prepare_nordland.py`) | `dataset` |
| 2 | SIFT + RANSAC (`src/methods/sift_ransac.py`) | `sift-baseline` |
| 3 | SALAD (`src/methods/salad.py`) | `salad-baseline` |
| 4 | Integration + evaluation (`src/evaluation`, `scripts/`) | `eval` |

- Branch off `main`, open a PR to merge back. Keep `main` runnable (`pytest` passes).
- Changes to the shared contracts (split format, rankings format, `RetrievalMethod`, timing stages)
  need a heads-up to the whole team and an entry in `docs/DECISIONS.md`.
- Never commit images, caches, or model weights (`.gitignore` covers `data/`, `cache/`, `*.npy`, `*.ckpt`).
