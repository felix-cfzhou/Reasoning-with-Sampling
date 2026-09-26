# Reasoning with Sampling: Cutting at Decision Points

Code for [*Reasoning with Sampling: Cutting at Decision Points*](https://arxiv.org/abs/2605.30327)
(NeurIPS 2026). The repository implements **Entropy-Cut Metropolis–Hastings**, a sampler for the
power distribution `p^α` of a base model that places MCMC cuts at positions of high next-token
entropy increase rather than uniformly. It also contains the paper's baselines: standard and
low-temperature sampling, uniform-cut MH ([Karan and Du](https://github.com/aakaran/reasoning-with-sampling)),
SMC, TMC and compute-matched Best-of-N. Evaluation covers MATH500, HumanEval, GPQA Diamond and AIME26.
Everything runs on a single GPU with vLLM.


## Repository layout

```
llm_experiments/
  power_sampling/   sampler library
    base.py           BaseSampler: loads the vLLM model; batch_naive_temp (Standard / Low-Temperature)
    mcmc_cut.py       McmcCutSampler   (Entropy-Cut MH)
    mcmc_orig.py      McmcOrigSampler  (Uniform-Cut MH)
    smc.py            SmcSampler       (SMC)
    tmc.py            TmcSampler       (TMC)
    best_of_n.py      BestOfNSampler   (compute-matched Standard / Low-Temperature BoN)
    outputs.py        result containers (NaiveTempOutput, McmcOutput, McmcHistory, ...)
    tracker.py        LogitsTracker (raw log-probs / entropies from the vLLM logits)
    prompts.py, constants.py, bon_utils.py, utils.py
  experiments/      one driver per benchmark (+ the suffix-divergence study)
  evaluation/       graders for the drivers' CSVs, clustered McNemar tests
  grader_utils/     answer parsing and grading helpers (MATH, GPQA, HumanEval execution)
  scripts/          single-GPU run scripts that reproduce the paper's experiments
```

Every sampler takes a `BaseSampler` in its constructor, so the model is loaded once and shared by
all arms:

```python
import os
os.environ["VLLM_USE_V1"] = "0"  # before importing vLLM

from power_sampling.base import BaseSampler
from power_sampling.mcmc_cut import McmcCutSampler

base = BaseSampler("Qwen/Qwen2.5-Math-7B", max_model_len=4096)
ids = base.tokenizer(prompts).input_ids
naive = base.batch_naive_temp(batch_context_ids=ids, temp=0.25, sample_len=3072, top_k=50)
cut = McmcCutSampler(base).batch_mcmc_power_sample(batch_prompt_ids=ids, temp=0.25, mcmc_steps=10,
                                                   max_new_tokens=3072, cut_power=4.0, top_k=50)
```

`VLLM_USE_V1=0` must be set before vLLM is imported (every driver does this at the top):
`LogitsTracker` needs the V0 engine's per-request `logits_processors`.


## Setup

```bash
conda env create -f environment.yml
conda activate sampling_reasoning

# CUDA libraries and headers from the conda env
export CUDA_HOME=$CONDA_PREFIX
export CMAKE_PREFIX_PATH=$CONDA_PREFIX

# NVIDIA architectures to compile vLLM for
export TORCH_CUDA_ARCH_LIST="8.0,8.9,9.0,10.0"

# link the CUDA headers where the vLLM build expects them
ln -s $CONDA_PREFIX/targets/x86_64-linux/include/* $CONDA_PREFIX/include/

# build vLLM v0.9.2, the last release with the V0 engine, from source
cd ..
git clone https://github.com/vllm-project/vllm --branch v0.9.2 --depth 1
cd vllm
MAX_JOBS=15 NVCC_THREADS=2 uv pip install -v -e . --no-build-isolation

# optional: check the vLLM install
cd ../<this-repo>
python3 test_vllm.py
```


## Data

Datasets are not shipped. Everything runs from `llm_experiments/` and reads from `llm_experiments/data/`:

```bash
cd llm_experiments && mkdir -p data

# MATH500 (MIT)
curl -L -o data/MATH500.json https://raw.githubusercontent.com/aakaran/reasoning-with-sampling/main/llm_experiments/data/MATH500.json

# HumanEval (MIT)
curl -L https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz | gunzip > data/HumanEval.jsonl
```

AIME26 comes from [MathArena](https://huggingface.co/MathArena) (CC BY-NC-SA 4.0). GPQA Diamond is
gated: accept its terms on [Hugging Face](https://huggingface.co/datasets/Idavidrein/gpqa) and
`huggingface-cli login` first. Keep the row order: answer choices are shuffled per row with the run's `--seed`.

```python
import json
from datasets import load_dataset

def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

dump(({"problem": r["problem"], "answer": r["answer"], "id": r["problem_idx"]}
      for r in load_dataset("MathArena/aime_2026", split="train")), "data/aime2026.jsonl")
dump(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"), "data/GPQA.jsonl")
```


## Reproducing the paper

Run the scripts from anywhere; they `cd` into `llm_experiments/`, run one job at a time on the
current GPU (pick it with `CUDA_VISIBLE_DEVICES`) and write CSVs to `llm_experiments/results/`.

| experiment | script (`llm_experiments/scripts/`) | driver (`experiments/`) |
|---|---|---|
| Main results, MATH500 | `run_math500.sh` | `power_samp_math_main.py` |
| Main results, HumanEval | `run_humaneval.sh` | `power_samp_he_main.py` |
| Main results, GPQA Diamond | `run_gpqa.sh` | `power_samp_gpqa_main.py` |
| Main results, AIME26 | `run_aime.sh` | `power_samp_aime_main.py` |
| Compute-matched Best-of-N | `run_bon.sh {math500,humaneval,aime26}` | same drivers |
| Suffix divergence | `run_suffix_divergence.sh` | `suffix_divergence_main.py` |

By default each main-results script runs all six samplers for all five models with 8 seeds (AIME26:
8 seeds × 8 in-process repeats = 64 runs). Settings are environment overrides, e.g. a quick check:

```bash
MODELS=qwen_math SEEDS=0 BATCH_SIZE=4 bash llm_experiments/scripts/run_math500.sh
```

Overrides: `MODELS`, `SAMPLERS`, `SEEDS`, `TEMP`, `MCMC_STEPS`, `CUT_POWER`, `NUM_REPEATS` (AIME26),
`SAVE_DIR` (default `results`) and `BATCH_SIZE`/`BATCH_IDX` to run a slice of the dataset.

**Names.** CLI names map to the paper as follows.

| `--model` | model | `--samplers` | method |
|---|---|---|---|
| `qwen` | Qwen2.5-7B | `std` | Standard |
| `qwen_math` | Qwen2.5-Math-7B | `naive_temp` | Low-Temperature |
| `qwen3_8b` | Qwen3-8B-Base | `smc` | SMC |
| `phi35_instruct` | Phi-3.5-mini-instruct | `tmc` | TMC |
| `phi4_instruct` | Phi-4-mini-instruct | `mcmc_orig` | Uniform-Cut MH |
| | | `mcmc_cut` | Entropy-Cut MH |
| | | `bon_std`, `bon_naive` | Standard / Low-Temperature BoN |

**Hyperparameters** are the drivers' defaults:

| | MATH500 | HumanEval | GPQA Diamond | AIME26 |
|---|---|---|---|---|
| max length T | 3072 | 3072 | 6144 | 6144 |
| block size B = T / 16 | 192 | 192 | 384 | 384 |
| α (`--temp` = 1/α) | 4 (0.25) | 5 (0.2) | 4 (0.25) | 4 (0.25) |

β (`--cut_power`) = 4, N_MCMC (`--mcmc_steps`) = 10, the proposal is the low-temperature
distribution with τ = 1/α. SMC uses N = 64 particles; TMC uses B = 192, K = 8 candidates and
M = 8 look-ahead completions per candidate.

**Compute-matched Best-of-N** runs Standard, Low-Temperature and Entropy-Cut MH,
then the two BoN arms, in one process. Each problem's budget is that run's Entropy-Cut MH
decode-token count (rejected proposals included), candidates are drawn until the budget is
reached (the crossing one is kept), and the candidate with the highest mean base-model
log-likelihood is returned. Each BoN arm prints a `budget realization` summary.

**Ablations** vary one setting on Qwen2.5-Math-7B / MATH500. Give each setting its own
`SAVE_DIR`, since the result filename does not encode β:

```bash
MODELS=qwen_math SAMPLERS=mcmc_cut TEMP=0.5   SAVE_DIR=results/ablate_alpha2 bash llm_experiments/scripts/run_math500.sh
MODELS=qwen_math SAMPLERS=mcmc_cut CUT_POWER=2 SAVE_DIR=results/ablate_beta2  bash llm_experiments/scripts/run_math500.sh
MODELS=qwen_math SAMPLERS=mcmc_cut MCMC_STEPS=5 SAVE_DIR=results/ablate_mh5  bash llm_experiments/scripts/run_math500.sh
```

The paper's decision-point audit used separate tooling and is not part of this repository;
`--store_history` records every MH proposal and its cut position if you want to inspect them.


## Evaluation

From `llm_experiments/`, pass the folder holding one benchmark's result CSVs (all seeds) as a
positional argument:

```bash
python -m evaluation.eval_math <dir>        # MATH500: accuracy, pass@k, per-question running time
python -m evaluation.eval_he <dir> he       # HumanEval; writes he_<sampler>.jsonl and executes the tests
python -m evaluation.eval_gpqa <dir>        # GPQA Diamond
python -m evaluation.eval_aime <dir>        # AIME26
```

HumanEval grading executes model-generated code; run it in a sandboxed environment. The MATH500
CSVs also record each sample's base-model log-likelihood (`*_sum_raw_logprobs`) and mean next-token
entropy (`*_avg_raw_entropies`, the negative of the average confidence).

Clustered McNemar tests of Entropy-Cut MH against the baselines, including the Best-of-N arms, on
one model's MATH500, HumanEval and AIME26 folders:

```bash
python -m evaluation.mcnemar_test --math500 <dir> --humaneval <dir> --aime <dir> --model qwen_math
```


## License

MIT; see [`LICENSE`](LICENSE). The graders in `llm_experiments/grader_utils/` include code adapted
from openai/human-eval, openai/prm800k and hendrycks/math, all MIT-licensed; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).


## Citation

```bibtex
@inproceedings{zhou2026reasoning,
  title         = {Reasoning with Sampling: Cutting at Decision Points},
  author        = {Zhou, Felix and Mehrotra, Anay and Liu, Quanquan C.},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026},
  eprint        = {2605.30327},
  archivePrefix = {arXiv}
}
```
