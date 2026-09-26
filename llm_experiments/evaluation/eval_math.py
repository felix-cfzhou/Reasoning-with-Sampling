import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from grader_utils.math_grader import grade_answer

# Paper name -> answer column written by power_samp_math_main.py, in display order.
SAMPLERS = {
    "Standard": "std_answer",
    "Low-Temperature": "naive_answer",
    "SMC": "smc_answer",
    "TMC": "tmc_answer",
    "Uniform-Cut MH": "mcmc_orig_answer",
    "Entropy-Cut MH": "mcmc_cut_answer",
    "Standard BoN": "bon_std_answer",
    "Low-Temperature BoN": "bon_naive_answer",
}

ELAPSED_TIME_COLS = {
    "Standard": "std_elapsed_time",
    "Low-Temperature": "naive_elapsed_time",
    "SMC": "smc_elapsed_time",
    "TMC": "tmc_elapsed_time",
    "Uniform-Cut MH": "mcmc_orig_elapsed_time",
    "Entropy-Cut MH": "mcmc_cut_elapsed_time",
    "Standard BoN": "bon_std_elapsed_time",
    "Low-Temperature BoN": "bon_naive_elapsed_time",
}


def pass_at_k(n, c, k):
    """Unbiased estimator for pass@k (Codex paper, Chen et al. 2021).

    n: total number of samples per problem
    c: number of correct samples
    k: k in pass@k
    Returns 1.0 when fewer than k samples are incorrect.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0


def eval_math(fname):
    print(fname)
    df = pd.read_csv(fname)
    total = len(df)
    # Only evaluate samplers whose columns exist in this CSV
    present = {s: col for s, col in SAMPLERS.items() if col in df.columns}
    correct = {s: 0 for s in present}
    no_answer = {s: 0 for s in present}

    for i in range(total):
        correct_ans = df["correct_answer"][i]
        for name, col in present.items():
            ans = df[col][i]
            correct[name] += safe_grade(ans, correct_ans)
            if pd.isna(ans):
                no_answer[name] += 1

    return correct, no_answer, total


def math_results(fnames):
    correct_total = {}
    no_ans_total = {}
    file_accuracies = {}
    total_per_sampler = {}
    overall_total = 0

    # Per-question correctness tracking for pass@k.
    # question_correct[sampler][question_id] = number of correct samples
    # question_n[sampler][question_id] = number of total samples
    question_correct: Dict[str, Dict[Any, int]] = {}
    question_n: Dict[str, Dict[Any, int]] = {}

    for fname in fnames:
        df = pd.read_csv(fname)
        correct, no_answer, n = eval_math(fname)
        denom_file = max(n, 1)
        for s in correct:
            correct_total[s] = correct_total.get(s, 0) + correct[s]
            no_ans_total[s] = no_ans_total.get(s, 0) + no_answer[s]
            file_accuracies.setdefault(s, []).append(correct[s] / denom_file)
            total_per_sampler[s] = total_per_sampler.get(s, 0) + n

        # Track per-question correctness across files
        present = {s: col for s, col in SAMPLERS.items() if col in df.columns}
        for i in range(len(df)):
            qid = df["id"][i] if "id" in df.columns else i
            correct_ans = df["correct_answer"][i]
            for s, col in present.items():
                question_correct.setdefault(s, {}).setdefault(qid, 0)
                question_n.setdefault(s, {}).setdefault(qid, 0)
                question_n[s][qid] += 1
                ans = df[col][i]
                question_correct[s][qid] += safe_grade(ans, correct_ans)

        overall_total += n

    all_samplers = [s for s in SAMPLERS if s in correct_total]
    acc = {s: correct_total[s] / max(total_per_sampler[s], 1) for s in all_samplers}
    answered_acc = {
        s: correct_total[s] / max(total_per_sampler[s] - no_ans_total[s], 1)
        for s in all_samplers
    }

    std_devs = {}
    for s in all_samplers:
        if len(file_accuracies[s]) > 1:
            std_devs[s] = pd.Series(file_accuracies[s]).std()
        else:
            std_devs[s] = 0.0

    print(f"Files evaluated: {len(fnames)}")
    print(f"Total questions (global): {overall_total}")
    for s in all_samplers:
        print(
            f"{s:20s} accuracy: {acc[s]:.3f} (std: {std_devs[s]:.3f}) "
            f"(no answer: {no_ans_total[s]}, answered acc: {answered_acc[s]:.3f}, total for sampler: {total_per_sampler[s]})"
        )

    # --- pass@k estimation (each file = one independent sample) ---
    n_files = len(fnames)
    if n_files >= 2:
        ks = [k for k in [1, 2, 3, 4, 5, 6, 7, 8] if k <= n_files]
        print(f"\npass@k estimates (n={n_files} independent runs):")
        header = f"{'Sampler':<21}" + "".join(f"{'pass@'+str(k):>12}" for k in ks)
        print(header)
        print("-" * len(header))
        for s in all_samplers:
            if s not in question_correct:
                continue
            row = f"{s:<21}"
            for k in ks:
                scores = [
                    pass_at_k(question_n[s][qid], question_correct[s][qid], k)
                    for qid in question_correct[s]
                ]
                row += f"{np.mean(scores):>11.4f} "
            print(row)
        print()

    # --- Average per-question running time ---
    time_sums = {}
    time_counts = {}
    for fname in fnames:
        df = pd.read_csv(fname)
        for s, col in ELAPSED_TIME_COLS.items():
            if col in df.columns:
                valid = df[col].dropna()
                time_sums[s] = time_sums.get(s, 0.0) + valid.sum()
                time_counts[s] = time_counts.get(s, 0) + len(valid)

    if time_counts:
        print("Average per-question running time (seconds):")
        for s in [s for s in ELAPSED_TIME_COLS if s in time_counts]:
            avg_time = time_sums[s] / max(time_counts[s], 1)
            print(f"  {s:20s} {avg_time:.2f}s  (n={time_counts[s]})")

    results = {}
    for s in all_samplers:
        results[f"{s}_acc"] = acc[s]
        results[f"{s}_std"] = std_devs[s]
        results[f"{s}_no_answer"] = no_ans_total[s]
        results[f"{s}_answered_acc"] = answered_acc[s]
        results[f"{s}_total"] = total_per_sampler[s]
    for s in time_counts:
        results[f"{s}_avg_time"] = time_sums[s] / max(time_counts[s], 1)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    math_results(fnames)
