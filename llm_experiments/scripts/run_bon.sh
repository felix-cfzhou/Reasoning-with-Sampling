#!/usr/bin/env bash
# Compute-matched Best-of-N on the current device.
#
#   bash scripts/run_bon.sh DATASET        # DATASET: math500 | humaneval | aime26
#
# For each model x seed, one process runs Standard, Low-Temperature and Entropy-Cut MH,
# then two Best-of-N arms -- bon_std (temperature 1) and bon_naive (temperature 1/alpha).
# Each problem's BoN budget is that run's Entropy-Cut MH decode-token count, rejected
# proposals included; candidates are drawn until the budget is reached (the crossing one
# is kept) and the one with the highest mean base-model log-likelihood is returned.
#
# Env overrides: MODELS (default: Qwen2.5-Math-7B and Qwen3-8B-Base), SEEDS, TEMP,
# MCMC_STEPS, CUT_POWER, NUM_REPEATS (aime26), SAVE_DIR, BATCH_SIZE / BATCH_IDX.
set -euo pipefail
cd "$(dirname "$0")/.."  # llm_experiments/: drivers read data/ and write results/ here

DATASET="${1:?usage: run_bon.sh DATASET   (DATASET: math500 | humaneval | aime26)}"
MODELS="${MODELS:-qwen_math qwen3_8b}"
MCMC_STEPS="${MCMC_STEPS:-10}"
CUT_POWER="${CUT_POWER:-4.0}"  # beta
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"

EXTRA=()
case "$DATASET" in
  math500)   DRIVER=power_samp_math_main; TEMP="${TEMP:-0.25}" ;;
  humaneval) DRIVER=power_samp_he_main;   TEMP="${TEMP:-0.2}" ;;
  aime26)    DRIVER=power_samp_aime_main; TEMP="${TEMP:-0.25}"
             EXTRA=(--num_repeats "${NUM_REPEATS:-8}") ;;
  *) echo "unknown DATASET '$DATASET' (want: math500 | humaneval | aime26)" >&2
     exit 2 ;;
esac

for MODEL in $MODELS; do
  for SEED in $SEEDS; do
    # One folder per (dataset, model): evaluation.mcnemar_test pairs runs within a folder.
    python -m "experiments.$DRIVER" \
      --model "$MODEL" \
      --samplers std naive_temp mcmc_cut bon_std bon_naive \
      --temp "$TEMP" \
      --mcmc_steps "$MCMC_STEPS" \
      --cut_power "$CUT_POWER" \
      --seed "$SEED" \
      --save_str "${SAVE_DIR:-results}/bon_${DATASET}_${MODEL}" \
      ${EXTRA[@]+"${EXTRA[@]}"} \
      ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} ${BATCH_IDX:+--batch_idx "$BATCH_IDX"}
  done
done
