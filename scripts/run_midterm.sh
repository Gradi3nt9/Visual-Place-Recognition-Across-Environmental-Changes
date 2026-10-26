#!/usr/bin/env bash
# Run every method on one split, then build the comparison table and retrieval figures.
#
#   scripts/run_midterm.sh nordland_smoke                 # correctness check
#   scripts/run_midterm.sh nordland_dev1 final            # midterm numbers (use a clean commit)
#
# Env overrides: METHODS="random salad" (subset of methods), SKIP_FIGURES=1, PYTHON=python3.10
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT="${1:?usage: scripts/run_midterm.sh <split> [tag]}"
TAG="${2:-}"
PY="${PYTHON:-python3}"
read -r -a METHODS <<< "${METHODS:-random sift_ransac salad}"
SUFFIX=""; TAG_ARG=()
if [ -n "$TAG" ]; then SUFFIX="__$TAG"; TAG_ARG=(--tag "$TAG"); fi

if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: uncommitted changes. Results will be marked '*'; do not report them as final."
fi

for m in "${METHODS[@]}"; do
  "$PY" scripts/run_retrieval.py --method "$m" --split "$SPLIT" ${TAG_ARG[@]+"${TAG_ARG[@]}"}
done

RANKINGS=()
for m in "${METHODS[@]}"; do RANKINGS+=("results/rankings/${SPLIT}__${m}${SUFFIX}.json"); done
"$PY" scripts/evaluate.py "${RANKINGS[@]}" --out "${SPLIT}${SUFFIX}__midterm"

# Figures compare the two real methods (first two in METHODS), if both were run.
if [ -z "${SKIP_FIGURES:-}" ] && [ "${#METHODS[@]}" -ge 2 ]; then
  "$PY" scripts/visualize_retrievals.py --split "$SPLIT" "${RANKINGS[0]}" "${RANKINGS[1]}"
fi
