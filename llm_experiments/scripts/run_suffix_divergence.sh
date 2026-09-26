#!/usr/bin/env bash
# Suffix divergence on MATH500: resample the suffix 16 times from 5
# top-decile and 5 bottom-decile entropy-jump positions and measure how much it diverges.
# Settings can be overridden from the environment, e.g.
#   MODEL=qwen3_8b NUM_RESAMPLES=8 bash scripts/run_suffix_divergence.sh
# BATCH_SIZE / BATCH_IDX select a slice of the dataset (default: all of it).
set -euo pipefail
cd "$(dirname "$0")/.."  # llm_experiments/: drivers read data/ and write results/ here

MODEL="${MODEL:-qwen_math}"
TEMP="${TEMP:-0.25}"
SEED="${SEED:-42}"
NUM_RESAMPLES="${NUM_RESAMPLES:-16}"
NUM_POSITIONS="${NUM_POSITIONS:-5}"
MIN_GAP="${MIN_GAP:-10}"

python -m experiments.suffix_divergence_main \
  --model "$MODEL" \
  --temp "$TEMP" \
  --seed "$SEED" \
  --num_resamples "$NUM_RESAMPLES" \
  --num_positions "$NUM_POSITIONS" \
  --min_gap "$MIN_GAP" \
  --save_str "${SAVE_DIR:-results}/suffix_divergence_${MODEL}" \
  ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} ${BATCH_IDX:+--batch_idx "$BATCH_IDX"}
