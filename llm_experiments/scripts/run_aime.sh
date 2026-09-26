#!/usr/bin/env bash
# AIME26 main results: the six samplers for each of the five models x 8 seeds, one run
# at a time on the current device.  Each run repeats the dataset NUM_REPEATS times in one
# process (8 seeds x 8 repeats = 64 runs) and writes one _rep{r}.csv per repeat.
# Settings can be overridden from the environment; a quick run:
#   MODELS=qwen_math SEEDS=0 NUM_REPEATS=1 BATCH_SIZE=4 bash scripts/run_aime.sh
# BATCH_SIZE / BATCH_IDX select a slice of the dataset (default: all of it).
set -euo pipefail
cd "$(dirname "$0")/.."  # llm_experiments/: drivers read data/ and write results/ here

MODELS="${MODELS:-qwen qwen_math qwen3_8b phi35_instruct phi4_instruct}"
SAMPLERS="${SAMPLERS:-std naive_temp smc tmc mcmc_orig mcmc_cut}"
TEMP="${TEMP:-0.25}"           # 1 / alpha
MCMC_STEPS="${MCMC_STEPS:-10}"
CUT_POWER="${CUT_POWER:-4.0}"  # beta
NUM_REPEATS="${NUM_REPEATS:-8}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"

for MODEL in $MODELS; do
  for SEED in $SEEDS; do
    python -m experiments.power_samp_aime_main \
      --model "$MODEL" \
      --samplers $SAMPLERS \
      --temp "$TEMP" \
      --mcmc_steps "$MCMC_STEPS" \
      --cut_power "$CUT_POWER" \
      --num_repeats "$NUM_REPEATS" \
      --seed "$SEED" \
      --save_str "${SAVE_DIR:-results}/aime_${MODEL}" \
      ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} ${BATCH_IDX:+--batch_idx "$BATCH_IDX"}
  done
done
