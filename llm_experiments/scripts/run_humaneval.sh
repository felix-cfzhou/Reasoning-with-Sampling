#!/usr/bin/env bash
# HumanEval main results: the six samplers for each of the five models x 8 seeds, one
# run at a time on the current device.  Settings can be overridden from the environment;
# a quick run:
#   MODELS=qwen_math SEEDS=0 BATCH_SIZE=4 bash scripts/run_humaneval.sh
# BATCH_SIZE / BATCH_IDX select a slice of the dataset (default: all of it).
set -euo pipefail
cd "$(dirname "$0")/.."  # llm_experiments/: drivers read data/ and write results/ here

MODELS="${MODELS:-qwen qwen_math qwen3_8b phi35_instruct phi4_instruct}"
SAMPLERS="${SAMPLERS:-std naive_temp smc tmc mcmc_orig mcmc_cut}"
TEMP="${TEMP:-0.2}"            # 1 / alpha, alpha = 5 on HumanEval
MCMC_STEPS="${MCMC_STEPS:-10}"
CUT_POWER="${CUT_POWER:-4.0}"  # beta
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"

for MODEL in $MODELS; do
  for SEED in $SEEDS; do
    python -m experiments.power_samp_he_main \
      --model "$MODEL" \
      --samplers $SAMPLERS \
      --temp "$TEMP" \
      --mcmc_steps "$MCMC_STEPS" \
      --cut_power "$CUT_POWER" \
      --seed "$SEED" \
      --save_str "${SAVE_DIR:-results}/humaneval_${MODEL}" \
      ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} ${BATCH_IDX:+--batch_idx "$BATCH_IDX"}
  done
done
