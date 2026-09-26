# Adapted from https://github.com/openai/human-eval (MIT License, Copyright (c) OpenAI (https://openai.com)).
# See THIRD_PARTY_NOTICES.md.

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools

import numpy as np
import tqdm
import gzip
import json
import os


from grader_utils.he_execute import check_correctness


def read_problems(evalset_file: str = "./HumanEval.jsonl") -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = "./HumanEval.jsonl",
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)
        sample_meta = {}

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            c_id = completion_id[task_id]
            file_idx = sample.get("file_idx", 0)
            sample_meta[(task_id, c_id)] = file_idx

            args = (problems[task_id], completion, timeout, c_id)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        if len(completion_id) != len(problems):
            print(f"Warning: {len(completion_id)}/{len(problems)} problems attempted.")

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    file_results = defaultdict(lambda: {"total": [], "correct": []})
    all_file_idxs = set(sample_meta.values()) if sample_meta else set([0])

    for task_id, result in results.items():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))

        # group by file_idx
        file_passed = defaultdict(list)
        for r in result:
            c_id = r[0]
            file_idx = sample_meta[(task_id, c_id)]
            file_passed[file_idx].append(r[1]["passed"])

        for file_idx in all_file_idxs:
            p_list = file_passed.get(file_idx, [])
            file_results[file_idx]["total"].append(len(p_list))
            file_results[file_idx]["correct"].append(sum(p_list))

    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # per-file pass at k
    file_pass_at_k = defaultdict(dict)
    for file_idx, fr in file_results.items():
        f_total = np.array(fr["total"])
        f_correct = np.array(fr["correct"])
        for k_val in ks:
            if (f_total >= k_val).all():
                file_pass_at_k[file_idx][f"pass@{k_val}"] = estimate_pass_at_k(
                    f_total, f_correct, k_val
                ).mean()

    # compute mean and std across files
    if len(file_pass_at_k) > 1:
        for k_val in ks:
            k_str = f"pass@{k_val}"
            vals = [
                file_pass_at_k[f_idx].get(k_str, float("nan"))
                for f_idx in file_pass_at_k
            ]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) > 0:
                pass_at_k[f"{k_str}_mean"] = np.mean(vals)
                pass_at_k[f"{k_str}_std"] = (
                    np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                )

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k
