"""Clustered McNemar tests of Entropy-Cut MH against the baselines.

Compares ``mcmc_cut`` (Entropy-Cut MH) against ``std`` (Standard), ``naive``
(Low-Temperature), ``mcmc_orig`` (Uniform-Cut MH) and, when present, the
compute-matched Best-of-N arms ``bon_std`` / ``bon_naive`` on MATH500,
HumanEval and AIME26.

Every sampler in a result CSV was run on the same questions in the same process, so the
outcomes are paired run by run.  For each question i, b_i counts the runs where
Entropy-Cut MH is correct and the baseline is not, and c_i the reverse.  The runs of a
question form one cluster (Durkalski et al. 2003):

    a_i = b_i - c_i,        z = sum_i a_i / sqrt(sum_i a_i^2),

compared against a standard normal for a two-sided p-value.  No correction for
multiple comparisons is applied.

Usage (from llm_experiments/)::

    python -m evaluation.mcnemar_test --math500 results/bon_math500_qwen_math \\
                                      --humaneval results/bon_humaneval_qwen_math \\
                                      --aime results/bon_aime26_qwen_math --model qwen_math

    python -m evaluation.mcnemar_test --selftest
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Graders are imported lazily inside the grading functions: they pull in sympy and
# pylatexenc, and --selftest exercises only the statistics, so it should run anywhere.

# Sampler column prefixes.  "naive" is low-temperature sampling, "std" is temperature 1.0.
ALGORITHM = "mcmc_cut"
BASELINES = ["std", "naive", "mcmc_orig", "bon_std", "bon_naive"]
SAMPLERS = [ALGORITHM] + BASELINES
PAPER_NAMES = {
    "mcmc_cut": "Entropy-Cut MH",
    "std": "Standard",
    "naive": "Low-Temperature",
    "mcmc_orig": "Uniform-Cut MH",
    "bon_std": "Standard BoN",
    "bon_naive": "Low-Temperature BoN",
}

# {model}_{bench}_base_power_samp_results_{steps}_{temp}_{batch_idx}_{seed}[_rep{r}].csv
# The model name can itself end in the bench token (qwen_math_math_...), so the model
# group is non-greedy and the bench alternation is anchored by the literal that follows.
RESULT_RE = re.compile(
    r"^(?P<model>.+?)_(?P<bench>math|aime|gpqa|he)"
    r"_base_power_samp_results_"
    r"(?P<steps>\d+)_(?P<temp>[0-9]*\.?[0-9]+)_"
    r"(?P<batch_idx>\d+)_(?P<seed>\d+)"
    r"(?:_rep(?P<rep>\d+))?\.csv$"
)

BENCH_INFO = {
    # bench key -> (filename token, expected question count)
    "math500": ("math", 500),
    "humaneval": ("he", 164),
    "aime": ("aime", 30),
}


# --------------------------------------------------------------------------- graders


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


_GRADE_CACHE = {}
_GRADE_STATS = defaultdict(int)


def _clean(value):
    """Normalise a CSV cell to a string, or None when there is no answer."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _clean_code(value):
    """Like _clean but preserves whitespace — stripping a Python body's leading
    indentation turns it into an IndentationError.  Matches eval_he.fnames_to_json.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def grade_math(answer, gold, timeout_s=5):
    """grade_answer with memoisation and a hang guard.

    math_grader warns that sympy can hang; the bare excepts in eval_math.safe_grade
    catch exceptions but not hangs, so wrap each call in an alarm as well.  Answers
    repeat heavily across seeds, so the cache removes most of the work.
    """
    from grader_utils.math_grader import grade_answer

    answer = _clean(answer)
    if answer is None:
        return 0
    # AIME golds arrive as numpy ints and normalize_answer calls .strip() before its
    # own try block, so coerce first — eval_aime.safe_grade_aime does the same.
    gold = _clean(gold)
    if gold is None:
        return 0
    key = (answer, gold)
    if key in _GRADE_CACHE:
        return _GRADE_CACHE[key]
    prev = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        result = int(bool(grade_answer(answer, gold)))
    except _Timeout:
        _GRADE_STATS["timeout"] += 1
        result = 0
    except Exception:
        _GRADE_STATS["exception"] += 1
        result = 0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
    _GRADE_CACHE[key] = result
    return result


def grade_aime(answer, gold):
    """AIME grading: sympy normalisation, then a strict integer fallback.

    Mirrors eval_aime.safe_grade_aime so these numbers stay comparable to what the
    existing aggregator reports.
    """
    if grade_math(answer, gold):
        return 1
    answer = _clean(answer)
    if answer is None:
        return 0
    try:
        return int(int(float(answer)) == int(float(str(gold))))
    except (ValueError, TypeError):
        return 0


# --------------------------------------------------------------------- file discovery


def _is_degenerate_code(code):
    """True when an extracted body carries no executable statement.

    extract_code captures up to the first line not indented by >=2 spaces, so a
    completion whose first line is indented by a single space is truncated down to the
    prompt's own docstring.  That silently scores 0 on every test.
    """
    body = code.strip()
    if not body:
        return True
    try:
        import ast

        tree = ast.parse("def _f():\n" + "\n".join("    " + l for l in body.splitlines()))
    except SyntaxError:
        return False  # broken code is a real failure, not an extraction artifact
    statements = tree.body[0].body
    return all(
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        for node in statements
    )


def parse_result_filename(path):
    match = RESULT_RE.match(Path(path).name)
    if match is None:
        return None
    fields = match.groupdict()
    return {
        "path": str(path),
        "model": fields["model"],
        "bench": fields["bench"],
        "steps": int(fields["steps"]),
        "temp": float(fields["temp"]),
        "batch_idx": int(fields["batch_idx"]),
        "seed": int(fields["seed"]),
        "rep": int(fields["rep"]) if fields["rep"] is not None else 0,
    }


def select_run_files(folder, bench, model=None):
    """Find the result CSVs for one benchmark, resolving folders that mix configs.

    A folder reused across runs can hold several configs (e.g. mcmc_steps=10 and 20),
    so group by config, warn, and keep the largest group.
    """
    token = BENCH_INFO[bench][0]
    runs = []
    unparsed = []
    for path in sorted(Path(folder).glob("*.csv")):
        info = parse_result_filename(path)
        if info is None:
            unparsed.append(path.name)
            continue
        if info["bench"] != token:
            continue
        if model is not None and info["model"] != model:
            continue
        runs.append(info)

    if unparsed:
        warnings.warn(
            f"{folder}: ignoring {len(unparsed)} file(s) whose names do not parse, "
            f"e.g. {unparsed[0]}"
        )
    if not runs:
        raise SystemExit(
            f"{folder}: no {bench} CSVs"
            + (f" for model {model!r}" if model else "")
            + ". Check the folder and --model."
        )

    groups = defaultdict(list)
    for info in runs:
        groups[(info["model"], info["steps"], info["temp"], info["batch_idx"])].append(info)
    if len(groups) > 1:
        print(f"  WARNING: {folder} holds {len(groups)} configurations:")
        for (mdl, steps, temp, batch), members in sorted(groups.items()):
            print(
                f"    model={mdl} mcmc_steps={steps} temp={temp} batch_idx={batch}"
                f"  ({len(members)} files)"
            )
        print("    using the largest group; pass --model to disambiguate")
    config = max(groups, key=lambda k: len(groups[k]))
    chosen = sorted(groups[config], key=lambda info: (info["seed"], info["rep"]))
    return chosen, dict(zip(("model", "steps", "temp", "batch_idx"), config))


# ------------------------------------------------------------------------- outcomes


class Outcomes:
    """Per-sampler (n_questions, n_runs) correctness matrices.

    ``runs[sampler]`` lists the (seed, rep) id of each matrix column, so two samplers
    can be paired run-by-run even when they appear in different subsets of files.
    """

    def __init__(self, bench, config, run_ids, n_questions):
        self.bench = bench
        self.config = config
        self.run_ids = run_ids
        self.n_questions = n_questions
        self.correct = {}
        self.runs = {}

    def add(self, sampler, correct, runs):
        self.correct[sampler] = np.asarray(correct, dtype=np.int8)
        self.runs[sampler] = list(runs)


def _grade_frame(df, bench, samplers):
    """Grade one MATH500 / AIME26 result CSV.  Returns {sampler: correct[]}."""
    grader = grade_aime if bench == "aime" else grade_math
    out = {}
    for sampler in samplers:
        column = f"{sampler}_answer"
        if column not in df.columns:
            continue
        out[sampler] = [grader(v, g) for v, g in zip(df[column], df["correct_answer"])]
    return out


def load_outcomes(folder, bench, model=None, samplers=SAMPLERS):
    """Grade every run CSV for one benchmark into aligned (question, run) matrices."""
    runs, config = select_run_files(folder, bench, model)
    expected_n = BENCH_INFO[bench][1]

    per_run = []
    for info in runs:
        df = pd.read_csv(info["path"])
        if len(df) != expected_n:
            raise SystemExit(
                f"{info['path']}: {len(df)} rows, expected {expected_n} for {bench}"
            )
        per_run.append(_grade_frame(df, bench, samplers))

    run_ids = [(info["seed"], info["rep"]) for info in runs]
    if len(set(run_ids)) != len(run_ids):
        raise SystemExit(
            f"{folder}: duplicate (seed, rep) across files — a folder was copied or "
            f"globbed twice. Refusing to double-count."
        )

    outcomes = Outcomes(bench, config, run_ids, expected_n)
    for sampler in samplers:
        present = [(run_id, g) for run_id, g in zip(run_ids, per_run) if sampler in g]
        if not present:
            continue
        if len(present) != len(per_run):
            print(
                f"  WARNING: {sampler} present in {len(present)}/{len(per_run)} files; "
                f"using only those runs"
            )
        correct = np.array([g[sampler] for _, g in present], dtype=np.int8).T
        outcomes.add(sampler, correct, [run_id for run_id, _ in present])
    return outcomes


def load_humaneval(folder, model=None, samplers=SAMPLERS, data_file="data/HumanEval.jsonl",
                   workdir="mcnemar_he", n_workers=8, timeout=3.0):
    """Grade HumanEval by execution, reusing he_grader/he_check unchanged."""
    from grader_utils.he_grader import entry_point as he_entry_point, extract_code

    runs, config = select_run_files(folder, "humaneval", model)
    problem_file = str(Path(data_file).resolve())
    with open(data_file, "r", encoding="utf-8") as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]
    by_task = {d["task_id"]: d for d in dataset}
    task_order = {d["task_id"]: i for i, d in enumerate(dataset)}

    os.makedirs(workdir, exist_ok=True)
    outcomes = Outcomes("humaneval", config,
                        [(r["seed"], r["rep"]) for r in runs], len(dataset))

    for sampler in samplers:
        column = f"{sampler}_completion"
        frames = [(i, pd.read_csv(r["path"])) for i, r in enumerate(runs)]
        frames = [(i, df) for i, df in frames if column in df.columns]
        if not frames:
            continue

        sample_file = os.path.join(workdir, f"{sampler}.jsonl")
        degenerate = 0
        total = 0
        with open(sample_file, "w") as out:
            for file_idx, df in frames:
                for i in range(len(df)):
                    task_id = df["id"][i]
                    if task_id not in by_task:
                        raise SystemExit(f"unknown task_id {task_id!r} in run {file_idx}")
                    completion = _clean_code(df[column][i])
                    response = by_task[task_id]["prompt"] + completion
                    code = extract_code(response, by_task[task_id]["entry_point"])
                    total += 1
                    if _is_degenerate_code(code):
                        degenerate += 1
                    out.write(json.dumps({
                        "task_id": task_id,
                        "completion": code,
                        "file_idx": file_idx,
                    }) + "\n")
        if degenerate:
            print(
                f"  WARNING: {sampler}: extract_code returned a docstring-only or empty "
                f"body for {degenerate}/{total} completions. extract_code stops at the "
                f"first line not indented by >=2 spaces, so completions whose first line "
                f"is indented by 1 space get truncated to the prompt's docstring and "
                f"score 0. Check the raw completions before trusting these numbers."
            )

        # k="1" keeps he_check's internal pass@k cheap; we only need the per-sample flags.
        he_entry_point(sample_file, k="1", n_workers=n_workers, timeout=timeout,
                       problem_file=problem_file)

        correct = np.zeros((len(dataset), len(frames)), dtype=np.int8)
        seen = np.zeros_like(correct, dtype=bool)
        column_of = {file_idx: col for col, (file_idx, _) in enumerate(frames)}
        with open(sample_file + "_results.jsonl") as handle:
            for line in handle:
                record = json.loads(line)
                row = task_order[record["task_id"]]
                col = column_of[record["file_idx"]]
                correct[row, col] = int(bool(record["passed"]))
                seen[row, col] = True
        if not seen.all():
            raise SystemExit(
                f"{sampler}: {int((~seen).sum())} (task, run) cells missing from "
                f"{sample_file}_results.jsonl"
            )
        outcomes.add(sampler, correct, [(runs[i]["seed"], runs[i]["rep"]) for i, _ in frames])
    return outcomes


# ---------------------------------------------------------------------- statistics


def mcnemar_clustered(y_treat, y_control):
    """Clustered McNemar test (Durkalski et al. 2003) on (question, run) outcomes.

    a_i = b_i - c_i is the net discordance of question i over its runs, and
    z = sum_i a_i / sqrt(sum_i a_i^2).  Returns (z, two-sided p, log10 p).
    """
    a = (y_treat.astype(np.int32) - y_control.astype(np.int32)).sum(axis=1)
    denominator = math.sqrt(float((a.astype(np.int64) ** 2).sum()))
    if denominator == 0:
        return 0.0, 1.0, 0.0
    z = float(a.sum()) / denominator
    # erfc is stable where 2*(1-Phi(|z|)) would underflow via the CDF.
    p = math.erfc(abs(z) / math.sqrt(2.0))
    log10_p = math.log10(p) if p > 0 else -(z * z / 2.0) / math.log(10.0)
    return z, p, log10_p


def contrast(outcomes, baseline):
    """Compare the algorithm against one baseline on a loaded benchmark."""
    treat = outcomes.correct[ALGORITHM]
    control = outcomes.correct[baseline]
    treat_runs = outcomes.runs[ALGORITHM]
    control_runs = outcomes.runs[baseline]
    if treat_runs != control_runs:
        # Different run coverage: pair only the (seed, rep) runs both arms have, so
        # every compared cell comes from the same process on the same question.
        control_col = {run: j for j, run in enumerate(control_runs)}
        shared = [(i, control_col[run]) for i, run in enumerate(treat_runs)
                  if run in control_col]
        print(
            f"  WARNING: {ALGORITHM} has {len(treat_runs)} runs and {baseline} has "
            f"{len(control_runs)}; pairing on the {len(shared)} runs they share"
        )
        if not shared:
            return None
        treat = treat[:, [i for i, _ in shared]]
        control = control[:, [j for _, j in shared]]

    if not ((treat == 1) != (control == 1)).any():
        # Degenerate (often both arms scoring zero). Skip this contrast loudly rather
        # than aborting, so the other benchmarks still produce a table.
        print(
            f"  WARNING: {outcomes.bench} vs {baseline}: no discordant pairs "
            f"(acc_cut={treat.mean():.3f}, acc_{baseline}={control.mean():.3f}) — "
            f"skipping this contrast"
        )
        return None
    z, p, log10_p = mcnemar_clustered(treat, control)
    return {
        "bench": outcomes.bench,
        "baseline": baseline,
        "acc_cut": float(treat.mean()),
        "acc_base": float(control.mean()),
        "n_questions": int(treat.shape[0]),
        "n_runs": int(treat.shape[1]),
        "z": z,
        "p": p,
        "log10_p": log10_p,
    }


# -------------------------------------------------------------------------- output


def fmt_p(p, log10_p):
    if log10_p < -300:
        return f"1e{log10_p:.0f}"
    if p >= 1e-4:
        return f"{p:.4f}"
    return f"{p:.1e}"


def report(rows):
    header = (
        f"{'bench':<11}{'baseline':<21}{'score':>8}{'Entropy-Cut':>13}"
        f"{'N':>5}{'R':>4}{'z':>8}{'p-value':>11}"
    )
    print("\n" + "=" * len(header))
    print(f"Clustered McNemar: {PAPER_NAMES[ALGORITHM]} vs baselines ({len(rows)} tests)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['bench']:<11}{PAPER_NAMES[row['baseline']]:<21}"
            f"{100 * row['acc_base']:>8.1f}{100 * row['acc_cut']:>13.1f}"
            f"{row['n_questions']:>5}{row['n_runs']:>4}{row['z']:>8.2f}"
            f"{fmt_p(row['p'], row['log10_p']):>11}"
        )
    print("-" * len(header))
    print(
        "Scores (%) are over the paired runs used for each test.  N = questions\n"
        "(clusters), R = paired runs.  Two-sided p-values, no multiple-comparison correction."
    )
    return rows


# ------------------------------------------------------------------------ self-test


def selftest():
    failures = []

    def check(name, condition, detail=""):
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not condition:
            failures.append(name)

    print("1. closed-form checks")
    # Three questions x 2 runs with net discordances a = (2, -1, 1): z = 2 / sqrt(6).
    treat = np.array([[1, 1], [0, 0], [1, 0]], dtype=np.int8)
    control = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.int8)
    z, p, _ = mcnemar_clustered(treat, control)
    z_expected = 2 / math.sqrt(6)
    check("z = sum(a) / sqrt(sum(a^2))", abs(z - z_expected) < 1e-12, f"z={z:.6f}")
    check("two-sided p", abs(p - math.erfc(z_expected / math.sqrt(2))) < 1e-12,
          f"p={p:.6f}")
    z0, p0, _ = mcnemar_clustered(treat, treat)
    check("identical arms -> z = 0, p = 1", z0 == 0.0 and p0 == 1.0)
    z_sym, _, _ = mcnemar_clustered(control, treat)
    check("antisymmetric in the two arms", abs(z_sym + z) < 1e-12)

    print("2. a question counts once, however many runs it has")
    # m questions where the algorithm wins all 8 runs: every a_i = 8, so z = sqrt(m)
    # regardless of the run count, unlike a test over independent (question, run) cells.
    m = 100
    treat = np.ones((m, 8), dtype=np.int8)
    control = np.zeros_like(treat)
    z, _, _ = mcnemar_clustered(treat, control)
    check("perfectly correlated runs give z = sqrt(#questions)",
          abs(z - math.sqrt(m)) < 1e-9, f"z={z:.3f}")

    print("3. calibration under the null (400 replicates, 200q x 8r)")
    rejections = 0
    trials = 400
    rng = np.random.default_rng(1)
    for _ in range(trials):
        rate = rng.uniform(0.2, 0.8, size=(200, 1))
        a = (rng.random((200, 8)) < rate).astype(np.int8)
        d = (rng.random((200, 8)) < rate).astype(np.int8)
        if mcnemar_clustered(a, d)[1] < 0.05:
            rejections += 1
    rate = rejections / trials
    check("clustered type-I error near 0.05", 0.025 <= rate <= 0.085, f"rate={rate:.3f}")

    print("4. power against a planted effect (200 replicates, +8pp)")
    rejections = 0
    trials = 200
    rng = np.random.default_rng(2)
    for _ in range(trials):
        rate = rng.uniform(0.2, 0.7, size=(200, 1))
        a = (rng.random((200, 8)) < np.clip(rate + 0.08, 0, 1)).astype(np.int8)
        d = (rng.random((200, 8)) < rate).astype(np.int8)
        if mcnemar_clustered(a, d)[1] < 0.05:
            rejections += 1
    rate = rejections / trials
    check("power at +8pp above 0.80", rate > 0.80, f"power={rate:.3f}")

    print("\n" + ("SELFTEST FAILED: " + ", ".join(failures) if failures else "selftest ok"))
    return 1 if failures else 0


# ------------------------------------------------------------------------------ cli


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--math500", help="folder of MATH500 result CSVs")
    parser.add_argument("--humaneval", help="folder of HumanEval result CSVs")
    parser.add_argument("--aime", help="folder of AIME26 result CSVs")
    parser.add_argument("--model", help="only use files whose filename model matches")
    parser.add_argument("--data-dir", default="data",
                        help="directory holding HumanEval.jsonl (for --humaneval)")
    parser.add_argument("--he-workers", type=int, default=8)
    parser.add_argument("--he-timeout", type=float, default=3.0)
    parser.add_argument("--he-workdir", default="mcnemar_he")
    parser.add_argument("--json-out", help="also write the results table as JSON")
    parser.add_argument("--selftest", action="store_true", help="run internal checks only")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    folders = {"math500": args.math500, "humaneval": args.humaneval, "aime": args.aime}
    if not any(folders.values()):
        parser.error("give at least one of --math500 --humaneval --aime")

    loaded = {}
    for bench, folder in folders.items():
        if folder is None:
            continue
        print(f"\nLoading {bench} from {folder}")
        if bench == "humaneval":
            outcomes = load_humaneval(
                folder, args.model,
                data_file=os.path.join(args.data_dir, "HumanEval.jsonl"),
                workdir=args.he_workdir, n_workers=args.he_workers,
                timeout=args.he_timeout,
            )
        else:
            outcomes = load_outcomes(folder, bench, args.model)
        missing = [s for s in SAMPLERS if s not in outcomes.correct]
        if missing:
            print(f"  WARNING: no columns for {', '.join(missing)} — skipping those")
        if ALGORITHM not in outcomes.correct:
            print(f"  ERROR: {ALGORITHM} not present in {folder}; skipping {bench}")
            continue
        cfg = outcomes.config
        print(f"  model={cfg['model']} mcmc_steps={cfg['steps']} temp={cfg['temp']} "
              f"runs={len(outcomes.run_ids)} questions={outcomes.n_questions}")
        loaded[bench] = outcomes

    if not loaded:
        raise SystemExit("nothing to test")

    rows = [
        row
        for bench, outcomes in loaded.items()
        for baseline in BASELINES
        if baseline in outcomes.correct
        for row in [contrast(outcomes, baseline)]
        if row is not None
    ]
    if not rows:
        raise SystemExit("no testable contrasts")
    report(rows)
    if _GRADE_STATS:
        print(f"\ngrader diagnostics: {dict(_GRADE_STATS)}")
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
