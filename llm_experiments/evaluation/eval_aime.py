import pandas as pd
import argparse
from pathlib import Path
from grader_utils.math_grader import grade_answer

# Paper name -> answer column written by power_samp_aime_main.py, in display order.
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


def safe_grade_aime(ans, correct_ans):
    """Grade an AIME answer using sympy-based normalization (like MATH500).

    Uses grade_answer which normalises both the solution and the reference
    answer via sympy before comparing, so fractional / radical / LaTeX
    expressions are handled correctly.  Falls back to direct integer comparison
    as a secondary check since AIME answers are always integers 0-999.
    """
    try:
        if pd.isna(ans):
            return 0
        ans_str = str(ans)
        correct_str = str(correct_ans)
        # Primary: sympy-based normalisation (same as MATH500)
        if grade_answer(ans_str, correct_str):
            return 1
        # Fallback: strict integer comparison
        return int(int(float(ans_str)) == int(float(correct_str)))
    except (ValueError, TypeError, Exception):
        return 0


def eval_aime(fname):
    print(fname)
    df = pd.read_csv(fname)
    total = len(df)
    present = {s: col for s, col in SAMPLERS.items() if col in df.columns}
    correct = {s: 0 for s in present}
    no_answer = {s: 0 for s in present}

    for i in range(total):
        correct_ans = df["correct_answer"][i]
        for name, col in present.items():
            ans = df[col][i]
            correct[name] += safe_grade_aime(ans, correct_ans)
            if pd.isna(ans):
                no_answer[name] += 1

    return correct, no_answer, total


def aime_results(fnames):
    correct_total = {}
    no_ans_total = {}
    file_accuracies = {}
    total_per_sampler = {}
    overall_total = 0

    for fname in fnames:
        correct, no_answer, n = eval_aime(fname)
        denom_file = max(n, 1)
        for s in correct:
            correct_total[s] = correct_total.get(s, 0) + correct[s]
            no_ans_total[s] = no_ans_total.get(s, 0) + no_answer[s]
            file_accuracies.setdefault(s, []).append(correct[s] / denom_file)
            total_per_sampler[s] = total_per_sampler.get(s, 0) + n
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

    results = {}
    for s in all_samplers:
        results[f"{s}_acc"] = acc[s]
        results[f"{s}_std"] = std_devs[s]
        results[f"{s}_no_answer"] = no_ans_total[s]
        results[f"{s}_answered_acc"] = answered_acc[s]
        results[f"{s}_total"] = total_per_sampler[s]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    aime_results(fnames)
