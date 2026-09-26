import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["TQDM_LEAVE"] = "0"


import json
import random
import argparse

import numpy as np
import pandas as pd
import torch

from power_sampling.base import BaseSampler
from power_sampling.best_of_n import BestOfNSampler
from power_sampling.mcmc_cut import McmcCutSampler
from power_sampling.mcmc_orig import McmcOrigSampler
from power_sampling.smc import SmcSampler
from power_sampling.tmc import TmcSampler
from power_sampling.bon_utils import BON_MAX_N, summarize_budget_realization

# --model choice -> Hugging Face model id (the paper's five models).
MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen3_8b": "Qwen/Qwen3-8B-Base",
    "phi35_instruct": "microsoft/Phi-3.5-mini-instruct",
    "phi4_instruct": "microsoft/Phi-4-mini-instruct",
}

# Maximum generation length T; block size B = T / num_blocks.
MAX_NEW_TOKENS = 3072

# The model is given the raw code stub, and generation stops at these.
STOP_WORDS = [
    "\nclass",
    "\ndef",
    "\n#",
    "\nif",
    "\nprint",
    "\nassert",
    "\nimport",
    "\nfrom",
    "\n```",
    "if __name__",
]


def strip_stop_words(completion):
    """Drop trailing whitespace and a trailing stop word from a decoded completion."""
    completion = completion.rstrip()
    for word in STOP_WORDS:
        if completion.endswith(word):
            completion = completion[: -len(word)]
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_str", action="store", type=str, default="results/", dest="save_str"
    )
    parser.add_argument(
        "--model",
        action="store",
        default="qwen",
        type=str,
        choices=list(MODEL_IDS),
    )
    parser.add_argument(
        "--temp", action="store", default=0.2, type=float, dest="temperature"
    )
    parser.add_argument("--mcmc_steps", action="store", type=int, default=10)
    parser.add_argument("--num_blocks", action="store", type=int, default=16)
    parser.add_argument("--top_k", action="store", type=int, default=50)
    parser.add_argument("--batch_size", action="store", type=int, default=164)
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--cut_power", action="store", type=float, default=4.0)
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["naive_temp", "std", "smc", "mcmc_orig", "mcmc_cut", "tmc"],
        choices=[
            "naive_temp",
            "std",
            "smc",
            "mcmc_orig",
            "mcmc_cut",
            "tmc",
            "bon_std",
            "bon_naive",
        ],
        help="List of sampling methods to run",
    )
    parser.add_argument(
        "--store_history",
        action="store",
        type=str,
        default="no",
        choices=["no", "entropy", "entropy_change"],
        help="History colormap mode: 'no' disables history, 'entropy' colors by token entropy, "
        "'entropy_change' colors by the increase in entropy from the previous proposal token.",
    )
    args = parser.parse_args()

    # Fail before the (slow) model load: the BoN budget is read from mcmc_cut in memory.
    bon_arms = [s for s in args.samplers if s.startswith("bon_")]
    if bon_arms and "mcmc_cut" not in args.samplers:
        parser.error(
            "bon needs a token budget: run it in the same process as mcmc_cut "
            "(--samplers mcmc_cut bon_std bon_naive)"
        )

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = args.model
    temp = args.temperature
    mcmc_steps = args.mcmc_steps
    top_k = args.top_k
    save_str = args.save_str

    os.makedirs(save_str, exist_ok=True)
    print(f"Results will be saved in {save_str}")

    model_str = MODEL_IDS[model]

    with open("data/HumanEval.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded HumanEval dataset ({len(dataset)} problems)")

    base = BaseSampler(model_str, max_model_len=4096, stop=STOP_WORDS)
    smc_sampler = SmcSampler(base)
    mcmc_orig_sampler = McmcOrigSampler(base)
    mcmc_cut_sampler = McmcCutSampler(base)
    bon_sampler = BestOfNSampler(base)
    tmc_sampler = TmcSampler(base)

    print(f"Loaded {model_str} model")

    batch_size = args.batch_size
    start = batch_size * args.batch_idx
    end = batch_size * (args.batch_idx + 1)
    subset = dataset[start:end]

    # --- Prepare all prompts: the raw code stub, for every model ---
    prompts = [data["prompt"] for data in subset]
    task_ids = [data["task_id"] for data in subset]

    print("Tokenizing inputs...")
    batch_input_ids = base.tokenizer(prompts).input_ids

    samplers_to_run = args.samplers

    # --- Batched naive_temp sampling (low temperature) ---
    naive_temp_results = None
    if "naive_temp" in samplers_to_run:
        print("Running batched naive_temp sampling...")
        naive_temp_results = base.batch_naive_temp(
            batch_context_ids=batch_input_ids,
            temp=temp,
            sample_len=MAX_NEW_TOKENS,
            top_k=top_k,
        )

    # --- Batched standard sampling (temperature=1.0) ---
    std_results = None
    if "std" in samplers_to_run:
        print("Running batched standard sampling...")
        std_results = base.batch_naive_temp(
            batch_context_ids=batch_input_ids,
            temp=1.0,
            sample_len=MAX_NEW_TOKENS,
            top_k=top_k,
        )

    # --- Batched SMC power sampling ---
    smc_results = None
    if "smc" in samplers_to_run:
        print("Running batched SMC power sampling...")
        smc_results = smc_sampler.batch_smc_power_sample(
            batch_prompt_ids=batch_input_ids,
            num_particles=64,
            max_new_tokens=MAX_NEW_TOKENS,
            temp=temp,
            top_k=top_k,
            block_size=32,
        )

    # --- Batched MCMC power sampling (uniform cuts, Karan and Du) ---
    mcmc_orig_results = None
    if "mcmc_orig" in samplers_to_run:
        print("Running batched MCMC power sampling (orig)...")
        mcmc_orig_results = mcmc_orig_sampler.batch_mcmc_orig_power_sample(
            batch_prompt_ids=batch_input_ids,
            temp=temp,
            mcmc_steps=mcmc_steps,
            max_new_tokens=MAX_NEW_TOKENS,
            num_blocks=args.num_blocks,
            top_k=top_k,
            store_history=args.store_history != "no",
        )

    # --- Batched entropy-cut MCMC power sampling ---
    mcmc_cut_results = None
    if "mcmc_cut" in samplers_to_run:
        print("Running batched entropy-cut MCMC power sampling...")
        mcmc_cut_results = mcmc_cut_sampler.batch_mcmc_power_sample(
            batch_prompt_ids=batch_input_ids,
            temp=temp,
            mcmc_steps=mcmc_steps,
            max_new_tokens=MAX_NEW_TOKENS,
            num_blocks=args.num_blocks,
            top_k=top_k,
            cut_power=args.cut_power,
            store_history=args.store_history != "no",
        )

    # --- Compute-matched Best-of-N (must follow mcmc_cut: it consumes its budget) ---
    bon_results = {}
    if bon_arms:
        token_budgets = [r.n_generated_tokens for r in mcmc_cut_results]
        assert all(b > 0 for b in token_budgets), "mcmc_cut reported zero generated tokens"
        print(
            f"BoN token budgets: median={int(np.median(token_budgets))} "
            f"min={min(token_budgets)} max={max(token_budgets)} "
            f"total={sum(token_budgets)}"
        )
        for arm in bon_arms:
            arm_temp = 1.0 if arm == "bon_std" else temp
            print(f"Running batched Best-of-N ({arm}, temp={arm_temp})...")
            bon_results[arm] = bon_sampler.batch_best_of_n(
                batch_prompt_ids=batch_input_ids,
                temp=arm_temp,
                max_new_tokens=MAX_NEW_TOKENS,
                token_budgets=token_budgets,
                top_k=top_k,
            )
            summary = summarize_budget_realization(
                [r.n_generated_tokens for r in bon_results[arm]],
                token_budgets,
                [r.n_candidates for r in bon_results[arm]],
                max_new_tokens=MAX_NEW_TOKENS,
                max_n=BON_MAX_N,
            )
            print(f"BoN {arm} budget realization: {summary}")
            for warning in summary["warnings"]:
                print(f"  WARNING: {arm}: {warning}")

    # --- Batched TMC power sampling ---
    tmc_results = None
    if "tmc" in samplers_to_run:
        print("Running batched TMC power sampling...")
        tmc_results = tmc_sampler.batch_tmc_power_sample(
            batch_prompt_ids=batch_input_ids,
            temp=temp,
            max_new_tokens=MAX_NEW_TOKENS,
            top_k=top_k,
        )

    # --- Batch-decode MCMC histories ---
    if args.store_history != "no":
        histories_to_decode = []
        if mcmc_orig_results is not None:
            histories_to_decode.extend(
                r.history for r in mcmc_orig_results if r.history is not None
            )
        if mcmc_cut_results is not None:
            histories_to_decode.extend(
                r.history for r in mcmc_cut_results if r.history is not None
            )
        if histories_to_decode:
            print("Decoding MCMC history tokens...")
            for history in histories_to_decode:
                history.batch_decode(base.tokenizer)

    # --- Decode all completions in batch ---
    print("Decoding output tokens...")
    naive_temp_completions = None
    if naive_temp_results is not None:
        naive_temp_completions = base.tokenizer.batch_decode(
            [naive_temp_results[i].output_ids for i in range(len(subset))],
            skip_special_tokens=True,
        )

    std_completions = None
    if std_results is not None:
        std_completions = base.tokenizer.batch_decode(
            [std_results[i].output_ids for i in range(len(subset))],
            skip_special_tokens=True,
        )

    smc_completions = None
    if smc_results is not None:
        smc_completions = base.tokenizer.batch_decode(
            [smc_results[i].sampled_particle.output_ids for i in range(len(subset))],
            skip_special_tokens=True,
        )

    mcmc_orig_completions = None
    if mcmc_orig_results is not None:
        mcmc_orig_completions = base.tokenizer.batch_decode(
            [mcmc_orig_results[i].output_ids for i in range(len(subset))],
            skip_special_tokens=True,
        )

    mcmc_cut_completions = None
    if mcmc_cut_results is not None:
        mcmc_cut_completions = base.tokenizer.batch_decode(
            [mcmc_cut_results[i].output_ids for i in range(len(subset))],
            skip_special_tokens=True,
        )

    bon_completions = {
        arm: base.tokenizer.batch_decode(
            [res[i].output_ids for i in range(len(subset))],
            skip_special_tokens=True,
        )
        for arm, res in bon_results.items()
    }

    tmc_completions = None
    if tmc_results is not None:
        tmc_completions = base.tokenizer.batch_decode(
            [tmc_results[i].output_ids for i in range(len(subset))],
            skip_special_tokens=True,
        )

    # --- Collect results ---
    if mcmc_cut_results is not None and len(mcmc_cut_results) > 0:
        cut_stats_0 = mcmc_cut_results[0].stats
        print(
            f"MCMC cut hyperparams: temp={cut_stats_0.temp}, top_k={cut_stats_0.top_k}, "
            f"cut_power={cut_stats_0.cut_power}"
        )
    results = []
    for i, (prompt, task_id) in enumerate(zip(prompts, task_ids)):
        print(f"--- Problem {i} ---")
        print(f"Input text:\n{prompt}")

        result_dict = {
            "question": prompt,
            "id": task_id,
        }

        if naive_temp_completions is not None:
            naive_temp_completion = strip_stop_words(naive_temp_completions[i])
            print(f"Naive completion:\n{naive_temp_completion}")
            result_dict["naive_completion"] = naive_temp_completion

        if std_completions is not None:
            std_completion = strip_stop_words(std_completions[i])
            print(f"Std completion:\n{std_completion}")
            result_dict["std_completion"] = std_completion

        if smc_completions is not None:
            smc_completion = strip_stop_words(smc_completions[i])
            print(f"SMC completion:\n{smc_completion}")
            result_dict["smc_completion"] = smc_completion

        if mcmc_orig_completions is not None:
            mcmc_orig_completion = strip_stop_words(mcmc_orig_completions[i])
            orig_stats = mcmc_orig_results[i].stats
            print(f"MCMC orig completion:\n{mcmc_orig_completion}")
            print(
                f"MCMC orig accepts: {orig_stats.accepts_per_block} / {orig_stats.attempts_per_block}"
            )
            if args.store_history != "no" and mcmc_orig_results[i].history is not None:
                print(
                    mcmc_orig_results[i].history.get_latex_str(mode=args.store_history)
                )
            result_dict["mcmc_orig_completion"] = mcmc_orig_completion

        if mcmc_cut_completions is not None:
            mcmc_cut_completion = strip_stop_words(mcmc_cut_completions[i])
            cut_stats = mcmc_cut_results[i].stats
            print(f"MCMC cut completion:\n{mcmc_cut_completion}")
            print(
                f"MCMC cut accepts: {cut_stats.accepts_per_block} / {cut_stats.attempts_per_block}"
            )
            print(
                f"MCMC cut power log ratio: {[f'{x:.4e}' for x in cut_stats.avg_power_log_ratio]}"
            )
            print(
                f"MCMC cut proposal log ratio: {[f'{x:.4e}' for x in cut_stats.avg_proposal_log_ratio]}"
            )
            print(
                f"MCMC cut cut log ratio: {[f'{x:.4e}' for x in cut_stats.avg_cut_log_ratio]}"
            )
            if args.store_history != "no" and mcmc_cut_results[i].history is not None:
                print(
                    mcmc_cut_results[i].history.get_latex_str(mode=args.store_history)
                )
            result_dict["mcmc_cut_completion"] = mcmc_cut_completion
            result_dict["mcmc_cut_total_tokens"] = int(
                mcmc_cut_results[i].n_generated_tokens
            )

        # One iteration per BoN arm.  No {arm}_answer column: eval_he.py grades by
        # executing the completion.  The same stop-word post-processing as every other
        # arm, so extract_code grades all arms alike.
        for arm, comps in bon_completions.items():
            bon_out = bon_results[arm][i]
            bon_completion = strip_stop_words(comps[i])
            print(
                f"{arm} completion:\n{bon_completion}\n"
                f"{arm} n={bon_out.n_candidates} "
                f"tokens={bon_out.n_generated_tokens}/{bon_out.token_budget} "
                f"avg_logprob={bon_out.avg_logprob:.4f} stop={bon_out.stop_reason}"
            )
            result_dict[f"{arm}_completion"] = bon_completion
            result_dict[f"{arm}_elapsed_time"] = bon_out.elapsed_time
            result_dict[f"{arm}_n_candidates"] = int(bon_out.n_candidates)
            result_dict[f"{arm}_total_tokens"] = int(bon_out.n_generated_tokens)

        if tmc_completions is not None:
            tmc_completion = strip_stop_words(tmc_completions[i])
            print(f"TMC completion:\n{tmc_completion}")
            result_dict["tmc_completion"] = tmc_completion

        results.append(result_dict)

    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(
            save_str,
            model
            + "_he_base_power_samp_results_"
            + str(mcmc_steps)
            + "_"
            + str(temp)
            + "_"
            + str(args.batch_idx)
            + "_"
            + str(args.seed)
            + ".csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
