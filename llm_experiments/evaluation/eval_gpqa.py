import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from grader_utils.gpqa_grader import parse_answer_gpqa

# Completion columns written by power_samp_gpqa_main.py, with paper names, in display order.
SAMPLER_COLS = [
    ("std_completion", "Standard"),
    ("naive_completion", "Low-Temperature"),
    ("smc_completion", "SMC"),
    ("tmc_completion", "TMC"),
    ("mcmc_orig_completion", "Uniform-Cut MH"),
    ("mcmc_cut_completion", "Entropy-Cut MH"),
]


def safe_grade(completion, correct_answer):
    """Return 1 if the parsed answer letter matches correct_answer, else 0."""
    try:
        parsed = parse_answer_gpqa(str(completion))
        return int(parsed == correct_answer)
    except Exception:
        return 0


def eval_gpqa(fname):
    """Grade a single results CSV, returning per-sampler correct counts and total."""
    df = pd.read_csv(fname)
    total = len(df)
    counts = {}
    for col, _ in SAMPLER_COLS:
        if col in df.columns:
            counts[col] = sum(
                safe_grade(df[col][i], df["correct_answer"][i]) for i in range(total)
            )
    return counts, total


def gpqa_results(fnames):
    """Aggregate results across multiple CSVs and print per-sampler accuracy."""
    total_counts = {}
    total_n = 0
    file_accs = {}

    for fname in fnames:
        counts, n = eval_gpqa(fname)
        total_n += n
        for col, correct in counts.items():
            total_counts[col] = total_counts.get(col, 0) + correct
            if col not in file_accs:
                file_accs[col] = []
            file_accs[col].append(correct / n)

    print(f"Total problems: {total_n}")
    accs = {}
    for col, label in SAMPLER_COLS:
        if col in total_counts:
            acc = total_counts[col] / total_n
            accs[col] = acc
            std = np.std(file_accs[col], ddof=1) if len(file_accs[col]) > 1 else 0.0
            print(
                f"{label:20s}: {acc:.3f} ± {std:.3f}  ({total_counts[col]}/{total_n})"
            )

    return accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing result CSVs")
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    gpqa_results(fnames)
