import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["TQDM_LEAVE"] = "0"


import json
import random
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import torch

from power_sampling.constants import GPQA_QUERY_TEMPLATE
from power_sampling.base import BaseSampler
from power_sampling.mcmc_cut import McmcCutSampler
from power_sampling.mcmc_orig import McmcOrigSampler
from power_sampling.smc import SmcSampler
from power_sampling.tmc import TmcSampler

# --model choice -> Hugging Face model id (the paper's five models).
MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen3_8b": "Qwen/Qwen3-8B-Base",
    "phi35_instruct": "microsoft/Phi-3.5-mini-instruct",
    "phi4_instruct": "microsoft/Phi-4-mini-instruct",
}

# Maximum generation length T; block size B = T / num_blocks.
MAX_NEW_TOKENS = 6144


def format_gpqa_prompt(data, model=None, tokenizer=None):
    """Format a GPQA problem into a prompt string with shuffled answer choices.

    Randomly shuffles the three incorrect answers, inserts the correct answer
    at a random position, and formats the result using GPQA_QUERY_TEMPLATE.
    For the Phi models, wraps the result in the model's chat template as a
    single user message.

    Args:
        data: dict — a single GPQA problem with keys 'Question',
            'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2',
            'Incorrect Answer 3'.
        model: str — model identifier (optional).
        tokenizer: HuggingFace tokenizer — required for the Phi models.

    Returns:
        input_text: str — the formatted prompt string ready for tokenization.
        answer: str — the correct answer letter ('A', 'B', 'C', or 'D').
    """
    choices = [
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    random.shuffle(choices)
    gold_index = random.randint(0, 3)
    choices.insert(gold_index, data["Correct Answer"])
    input_text = GPQA_QUERY_TEMPLATE.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Question=data["Question"],
    )
    answer = "ABCD"[gold_index]

    if model in ["phi35_instruct", "phi4_instruct"]:
        messages = [
            {"role": "user", "content": input_text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return input_text, answer


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
        "--temp", action="store", default=0.25, type=float, dest="temperature"
    )
    parser.add_argument("--mcmc_steps", action="store", type=int, default=10)
    parser.add_argument("--num_blocks", action="store", type=int, default=16)
    parser.add_argument("--top_k", action="store", type=int, default=50)
    parser.add_argument("--batch_size", action="store", type=int, default=198)
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--cut_power", action="store", type=float, default=4.0)
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["naive_temp", "std", "smc", "mcmc_orig", "mcmc_cut", "tmc"],
        choices=["naive_temp", "std", "smc", "mcmc_orig", "mcmc_cut", "tmc"],
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

    with open("data/GPQA.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded GPQA Diamond dataset ({len(dataset)} problems)")

    base = BaseSampler(model_str, max_model_len=8192)
    smc_sampler = SmcSampler(base)
    mcmc_orig_sampler = McmcOrigSampler(base)
    mcmc_cut_sampler = McmcCutSampler(base)
    tmc_sampler = TmcSampler(base)

    print(f"Loaded {model_str} model")

    batch_size = args.batch_size
    start = batch_size * args.batch_idx
    end = batch_size * (args.batch_idx + 1)
    subset = dataset[start:end]

    # --- Prepare all prompts ---
    formatted = [
        format_gpqa_prompt(data, model, base.tokenizer)
        for data in tqdm(subset, desc="Formatting prompts", leave=True)
    ]
    batch_input_texts = [f[0] for f in formatted]
    correct_answers = [f[1] for f in formatted]

    print("Tokenizing inputs...")
    batch_input_ids = base.tokenizer(batch_input_texts).input_ids

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
    for i, (input_text, answer) in enumerate(zip(batch_input_texts, correct_answers)):
        print(f"--- Problem {i} ---")

        result_dict = {
            "question": input_text,
            "correct_answer": answer,
        }

        if naive_temp_completions is not None:
            naive_temp_completion = naive_temp_completions[i]
            print(f"Naive completion length: {len(naive_temp_completion)}")
            result_dict["naive_completion"] = naive_temp_completion

        if std_completions is not None:
            std_completion = std_completions[i]
            print(f"Std completion length: {len(std_completion)}")
            result_dict["std_completion"] = std_completion

        if smc_completions is not None:
            smc_completion = smc_completions[i]
            print(f"SMC completion length: {len(smc_completion)}")
            result_dict["smc_completion"] = smc_completion

        if mcmc_orig_completions is not None:
            mcmc_orig_completion = mcmc_orig_completions[i]
            orig_stats = mcmc_orig_results[i].stats
            print(f"MCMC orig completion length: {len(mcmc_orig_completion)}")
            print(
                f"MCMC orig accepts: {orig_stats.accepts_per_block} / {orig_stats.attempts_per_block}"
            )
            if args.store_history != "no" and mcmc_orig_results[i].history is not None:
                print(
                    mcmc_orig_results[i].history.get_latex_str(mode=args.store_history)
                )
            result_dict["mcmc_orig_completion"] = mcmc_orig_completion

        if mcmc_cut_completions is not None:
            mcmc_cut_completion = mcmc_cut_completions[i]
            cut_stats = mcmc_cut_results[i].stats
            print(f"MCMC cut completion length: {len(mcmc_cut_completion)}")
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

        if tmc_completions is not None:
            tmc_completion = tmc_completions[i]
            print(f"TMC completion length: {len(tmc_completion)}")
            result_dict["tmc_completion"] = tmc_completion

        results.append(result_dict)

    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(
            save_str,
            model
            + "_gpqa_base_power_samp_results_"
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
