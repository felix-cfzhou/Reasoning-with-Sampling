import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["TQDM_LEAVE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import json
import random
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd
import torch

from grader_utils.parse_utils import parse_answer
from power_sampling.base import BaseSampler
from power_sampling.prompts import format_prompt

# --model choice -> Hugging Face model id (the paper's five models).
MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen3_8b": "Qwen/Qwen3-8B-Base",
    "phi35_instruct": "microsoft/Phi-3.5-mini-instruct",
    "phi4_instruct": "microsoft/Phi-4-mini-instruct",
}


def compute_delta_t(raw_entropies):
    """Compute per-token entropy change (Δ_t = H(t) - H(t-1), clipped at 0)."""
    deltas = np.diff(raw_entropies, prepend=0).clip(min=0)
    return deltas


def select_cut_positions(deltas, decile="top", num_positions=5, min_gap=10):
    """Select positions from the top or bottom decile of Δ_t values.

    Args:
        deltas: Per-token delta values.
        decile: "top" for top decile (high Δ), "bottom" for bottom decile (low Δ).
        num_positions: How many cut positions to select.
        min_gap: Minimum gap between selected positions (to avoid clustering).

    Returns:
        List of selected position indices.
    """
    n = len(deltas)
    if n < 20:
        return []

    # Skip first and last 5% of tokens to avoid edge effects
    margin = max(5, n // 20)
    candidate_indices = np.arange(margin, n - margin)
    candidate_deltas = deltas[candidate_indices]

    # Sort candidates by delta value
    if decile == "top":
        sorted_order = np.argsort(-candidate_deltas)  # descending
    else:
        sorted_order = np.argsort(candidate_deltas)  # ascending

    # Take from the top/bottom decile with minimum gap spacing
    decile_size = max(1, len(candidate_indices) // 10)
    decile_candidates = candidate_indices[sorted_order[:decile_size]]

    # Greedily select positions with minimum gap
    selected = []
    for pos in decile_candidates:
        if all(abs(pos - s) >= min_gap for s in selected):
            selected.append(int(pos))
        if len(selected) >= num_positions:
            break

    return selected


def compute_pairwise_edit_distance(sequences):
    """Compute average pairwise token-level edit distance between sequences."""
    n = len(sequences)
    if n < 2:
        return 0.0

    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = _edit_distance(sequences[i], sequences[j])
            total_dist += dist
            count += 1

    return total_dist / count if count > 0 else 0.0


def _edit_distance(seq1, seq2):
    """Levenshtein distance between two token sequences."""
    m, n = len(seq1), len(seq2)
    # Use two-row optimization for memory efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def main():
    parser = argparse.ArgumentParser(
        description="Suffix divergence under resampling experiment"
    )
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
    parser.add_argument("--top_k", action="store", type=int, default=50)
    parser.add_argument("--batch_size", action="store", type=int, default=500)
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument(
        "--num_resamples",
        action="store",
        type=int,
        default=16,
        help="Number of suffix resamples per cut position",
    )
    parser.add_argument(
        "--num_positions",
        action="store",
        type=int,
        default=5,
        help="Number of cut positions to select per decile",
    )
    parser.add_argument(
        "--min_gap",
        action="store",
        type=int,
        default=10,
        help="Minimum token gap between selected cut positions",
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
    top_k = args.top_k
    save_str = args.save_str
    num_resamples = args.num_resamples
    num_positions = args.num_positions

    os.makedirs(save_str, exist_ok=True)
    print(f"Results will be saved in {save_str}")

    model_str = MODEL_IDS[model]

    with open("data/MATH500.json", "r") as f:
        dataset = json.load(f)

    print(f"Loaded MATH500 dataset ({len(dataset)} problems)")

    base = BaseSampler(model_str, max_model_len=4096)
    print(f"Loaded {model_str} model")

    batch_size = args.batch_size
    start = batch_size * args.batch_idx
    end = batch_size * (args.batch_idx + 1)
    subset = dataset[start:end]

    questions = [data["prompt"] for data in subset]
    answers = [data["answer"] for data in subset]

    batch_input_texts = [
        format_prompt(question, model, base.tokenizer)
        for question in tqdm(questions, desc="Formatting prompts", leave=True)
    ]
    print("Tokenizing inputs...")
    batch_input_ids = base.tokenizer(batch_input_texts).input_ids

    # Step 1: Generate initial completions for all problems
    print("Generating initial completions...")
    initial_results = base.batch_naive_temp(
        batch_context_ids=batch_input_ids,
        temp=temp,
        sample_len=3072,
        top_k=top_k,
    )

    all_results = []

    for i, (question, answer) in enumerate(
        tqdm(
            zip(questions, answers),
            total=len(questions),
            desc="Processing problems",
        )
    ):
        init_result = initial_results[i]
        output_ids = init_result.output_ids
        raw_entropies = np.array(init_result.raw_entropies)
        input_ids = batch_input_ids[i]

        if len(output_ids) < 20:
            print(f"Problem {i}: output too short ({len(output_ids)} tokens), skipping")
            continue

        # Compute per-token Δ_t
        deltas = compute_delta_t(raw_entropies)

        # Select cut positions from top and bottom deciles
        top_positions = select_cut_positions(
            deltas, decile="top", num_positions=num_positions, min_gap=args.min_gap
        )
        bottom_positions = select_cut_positions(
            deltas, decile="bottom", num_positions=num_positions, min_gap=args.min_gap
        )

        if not top_positions or not bottom_positions:
            print(f"Problem {i}: not enough positions to select, skipping")
            continue

        initial_completion = base.tokenizer.decode(
            output_ids, skip_special_tokens=True
        )
        initial_answer = parse_answer(initial_completion)

        print(f"\n--- Problem {i} ---")
        print(f"Output length: {len(output_ids)} tokens")
        print(f"Top-decile Δ positions: {top_positions}")
        print(f"Bottom-decile Δ positions: {bottom_positions}")
        print(f"Top-decile Δ values: {[f'{deltas[p]:.4f}' for p in top_positions]}")
        print(
            f"Bottom-decile Δ values: {[f'{deltas[p]:.4f}' for p in bottom_positions]}"
        )

        # Collect all cut positions with metadata for a single batched call
        all_cuts = []  # list of (decile_label, cut_pos)
        for decile_label, positions in [
            ("top", top_positions),
            ("bottom", bottom_positions),
        ]:
            for cut_pos in positions:
                all_cuts.append((decile_label, cut_pos))

        # Build all prefix contexts and find max sample_len across positions
        all_prefix_ids = []
        sample_lens = []
        for _, cut_pos in all_cuts:
            prefix_ids = list(input_ids) + list(output_ids[:cut_pos])
            remaining_len = 192  # max(64, len(output_ids) - cut_pos + 32)
            # Repeat each prefix num_resamples times
            all_prefix_ids.extend([prefix_ids] * num_resamples)
            sample_lens.extend([remaining_len] * num_resamples)

        # Use the max sample_len so all sequences can be generated in one call
        max_sample_len = max(sample_lens)

        # Single batched call for all positions × all resamples
        all_resample_results = base.batch_naive_temp(
            batch_context_ids=all_prefix_ids,
            temp=temp,
            sample_len=max_sample_len,
            top_k=top_k,
            use_tqdm=False,
        )

        # Unpack results per cut position
        result_offset = 0
        for decile_label, cut_pos in all_cuts:
            resample_results = all_resample_results[
                result_offset : result_offset + num_resamples
            ]
            result_offset += num_resamples

            # Collect resampled suffix token sequences and answers
            suffix_token_seqs = []
            resampled_answers = []
            for r in resample_results:
                suffix_token_seqs.append(r.output_ids)
                full_completion = base.tokenizer.decode(
                    list(output_ids[:cut_pos]) + r.output_ids,
                    skip_special_tokens=True,
                )
                resampled_answers.append(parse_answer(full_completion))

            # Compute diversity metrics
            avg_edit_dist = compute_pairwise_edit_distance(suffix_token_seqs)
            distinct_answers = len(set(a for a in resampled_answers if a is not None))
            # Distinct-answer fraction: unique parsed final answers / number of resamples.
            distinct_answer_fraction = distinct_answers / num_resamples
            none_count = sum(1 for a in resampled_answers if a is None)

            # Normalized edit distance (by average suffix length)
            avg_suffix_len = np.mean([len(s) for s in suffix_token_seqs])
            norm_edit_dist = (
                avg_edit_dist / avg_suffix_len if avg_suffix_len > 0 else 0.0
            )

            result_dict = {
                "problem_idx": i,
                "question": question,
                "correct_answer": answer,
                "initial_answer": initial_answer,
                "output_length": len(output_ids),
                "decile": decile_label,
                "cut_position": cut_pos,
                "cut_position_frac": cut_pos / len(output_ids),
                "delta_t": float(deltas[cut_pos]),
                "entropy_at_cut": float(raw_entropies[cut_pos]),
                "num_resamples": num_resamples,
                "avg_pairwise_edit_distance": avg_edit_dist,
                "normalized_edit_distance": norm_edit_dist,
                "avg_suffix_length": avg_suffix_len,
                "distinct_answers": distinct_answers,
                "distinct_answer_fraction": distinct_answer_fraction,
                "none_answers": none_count,
                "resampled_answers": json.dumps(resampled_answers),
            }
            all_results.append(result_dict)

            print(
                f"  {decile_label} decile, pos {cut_pos} (Δ={deltas[cut_pos]:.4f}): "
                f"edit_dist={avg_edit_dist:.1f}, norm_edit={norm_edit_dist:.3f}, "
                f"distinct_answers={distinct_answers}, none={none_count}"
            )

    # Save results
    df = pd.DataFrame(all_results)
    out_fname = os.path.join(
        save_str,
        f"{model}_suffix_divergence_results"
        f"_{num_resamples}_{temp}_{args.batch_idx}_{seed}.csv",
    )
    df.to_csv(out_fname, index=False)
    print(f"\nSaved {len(all_results)} results to {out_fname}")

    # Print summary statistics
    if len(all_results) > 0:
        print("\n=== Summary ===")
        for decile_label in ["top", "bottom"]:
            subset_df = df[df["decile"] == decile_label]
            if len(subset_df) > 0:
                print(f"\n{decile_label.upper()} decile (high Δ = top):")
                print(f"  Mean Δ_t: {subset_df['delta_t'].mean():.4f}")
                print(
                    f"  Mean pairwise edit distance: {subset_df['avg_pairwise_edit_distance'].mean():.1f}"
                )
                print(
                    f"  Mean normalized edit distance: {subset_df['normalized_edit_distance'].mean():.4f}"
                )
                print(
                    f"  Mean distinct-answer fraction: {subset_df['distinct_answer_fraction'].mean():.4f}"
                )

        # Statistical comparison
        top_edit = df[df["decile"] == "top"]["normalized_edit_distance"]
        bottom_edit = df[df["decile"] == "bottom"]["normalized_edit_distance"]
        top_distinct = df[df["decile"] == "top"]["distinct_answer_fraction"]
        bottom_distinct = df[df["decile"] == "bottom"]["distinct_answer_fraction"]

        if len(top_edit) > 0 and len(bottom_edit) > 0:
            print(
                f"\n  Edit distance ratio (top/bottom): {top_edit.mean() / max(bottom_edit.mean(), 1e-8):.2f}x"
            )
            print(
                f"  Distinct-answer fraction ratio (top/bottom): {top_distinct.mean() / max(bottom_distinct.mean(), 1e-8):.2f}x"
            )


if __name__ == "__main__":
    main()
