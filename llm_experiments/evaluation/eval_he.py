import argparse

import pandas as pd
import json
from pathlib import Path
from grader_utils.he_grader import entry_point, extract_code

# Map sampler tag to CSV column name.
# All completions from the vLLM-based pipeline contain only the generated
# tokens (no prompt prefix), so the HumanEval prompt is prepended to each.
TAG_TO_COLUMN = {
    "std": "std_completion",
    "naive": "naive_completion",
    "smc": "smc_completion",
    "tmc": "tmc_completion",
    "mcmc_orig": "mcmc_orig_completion",
    "mcmc_cut": "mcmc_cut_completion",
    "bon_std": "bon_std_completion",
    "bon_naive": "bon_naive_completion",
}

# Paper names for the summary table.
TAG_NAMES = {
    "std": "Standard",
    "naive": "Low-Temperature",
    "smc": "SMC",
    "tmc": "TMC",
    "mcmc_orig": "Uniform-Cut MH",
    "mcmc_cut": "Entropy-Cut MH",
    "bon_std": "Standard BoN",
    "bon_naive": "Low-Temperature BoN",
}


def fnames_to_json(fnames, output_fname, tag, data_file="data/HumanEval.jsonl"):
    with open(data_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # Map task_id to its dataset entry so we don't rely on file ordering
    task_id_to_data = {d["task_id"]: d for d in dataset}

    col_name = TAG_TO_COLUMN.get(tag)
    if col_name is None:
        raise ValueError(
            f"Unknown tag: {tag}. Valid tags: {list(TAG_TO_COLUMN.keys())}"
        )

    output_file = output_fname + "_" + tag + ".jsonl"
    with open(output_file, "w") as fout:
        for idx in range(len(fnames)):
            fname = fnames[idx]
            print(fname)
            df = pd.read_csv(fname)

            if col_name not in df.columns:
                print(f"  Skipping {fname}: column '{col_name}' not found")
                continue

            for i in range(len(df)):
                task_id = df["id"][i]
                if task_id not in task_id_to_data:
                    print(f"  Warning: Unknown task_id '{task_id}' in {fname}")
                    continue

                ep = task_id_to_data[task_id]["entry_point"]
                prompt = task_id_to_data[task_id]["prompt"]

                completion = df[col_name][i]
                if pd.isna(completion):
                    print(f"  Warning: NaN completion for {task_id}, tag={tag}")
                    completion = ""

                response = prompt + str(completion)
                code_completion = extract_code(response, ep)

                line = {
                    "task_id": df["id"][i],
                    "completion": code_completion,
                    "file_idx": idx,
                }

                fout.write(json.dumps(line) + "\n")
    return output_file


def he_results(fnames, output_fname, tags=None):
    if tags is None:
        tags = list(TAG_TO_COLUMN.keys())

    # Only grade samplers that at least one CSV actually contains.
    columns = set()
    for fname in fnames:
        columns.update(pd.read_csv(fname, nrows=0).columns)
    for tag in [t for t in tags if TAG_TO_COLUMN[t] not in columns]:
        print(f"Skipping {tag}: no CSV has a '{TAG_TO_COLUMN[tag]}' column")
    tags = [t for t in tags if TAG_TO_COLUMN[t] in columns]

    all_results = {}
    for tag in tags:
        print(f"\n{'='*60}")
        print(f"  Evaluating sampler: {tag}")
        print(f"{'='*60}")
        output_file = fnames_to_json(fnames, output_fname, tag)
        results = entry_point(output_file, problem_file="data/HumanEval.jsonl")
        all_results[tag] = results

    # --- Print aggregated summary ---
    print(f"\n{'='*80}")
    print("  HumanEval Results Summary (Aggregated & Average)")
    print(f"{'='*80}")
    # Collect all k values (ignoring mean/std keys)
    all_ks = sorted(
        {
            k
            for r in all_results.values()
            for k in r.keys()
            if "mean" not in k and "std" not in k
        }
    )

    # Identify keys that also have mean/std present
    avg_ks = []
    for k in all_ks:
        if any(f"{k}_mean" in r for r in all_results.values()):
            avg_ks.append(k)

    header_cols = [f"{k:>12}" for k in all_ks]
    for k in avg_ks:
        header_cols.append(f"{k+' (avg±std)':>20}")

    header = f"{'Sampler':<21}" + "".join(header_cols)
    print(header)
    print("-" * len(header))
    for tag in tags:
        if tag in all_results:
            row = f"{TAG_NAMES[tag]:<21}"
            for k in all_ks:
                val = all_results[tag].get(k, float("nan"))
                row += f"{val:>11.4f} "
            for k in avg_ks:
                mean_val = all_results[tag].get(f"{k}_mean", float("nan"))
                std_val = all_results[tag].get(f"{k}_std", float("nan"))
                if not pd.isna(mean_val):
                    val_str = f"{mean_val:.3f}±{std_val:.3f}"
                    row += f"{val_str:>20} "
                else:
                    row += f"{'NaN':>20} "
            print(row)
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("output_fname", type=str)
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        choices=list(TAG_TO_COLUMN.keys()),
        help="Sampler tags to evaluate. Defaults to all available.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    he_results(fnames, args.output_fname, tags=args.tags)
