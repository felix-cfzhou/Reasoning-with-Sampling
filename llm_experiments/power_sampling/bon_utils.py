"""Pure arithmetic for compute-matched Best-of-N sampling.

Kept free of vllm/torch imports so it can be exercised on a machine with no GPU
(from llm_experiments/):

    python -m power_sampling.bon_utils

`power_sampling/best_of_n.py` imports `plan_round_sizes`; the MATH500, HumanEval and
AIME26 drivers import `summarize_budget_realization`.
"""

import sys

import numpy as np

# Safety cap on candidates per prompt; the token budget, not this cap, normally ends a run.
BON_MAX_N = 256


def plan_round_sizes(
    remaining,
    n_drawn,
    max_n,
    max_new_tokens,
    per_prompt_cap=16,
    max_concurrent=1024,
):
    """How many BoN candidates to draw for each prompt in the next round.

    Args:
        remaining: List[int] — budget[i] - spent[i]. <= 0 means the prompt is done.
        n_drawn: List[int] — candidates already drawn for the prompt.
        max_n: Hard cap on candidates per prompt.
        max_new_tokens: Per-candidate generation cap — the worst-case candidate length.
        per_prompt_cap: Cap on candidates drawn for one prompt in a single round.
        max_concurrent: Cap on total in-flight requests, to bound GPU-side memory
            (each request holds a LogitsTracker with a cloned dense logits tensor).

    Returns:
        want: List[int] — k[i] candidates to draw. k[i] == 0 means prompt i leaves
            the active set.

    Fan-out is sized by the WORST-CASE candidate length: k = floor(rem / max_new_tokens)
    guarantees k * max_new_tokens <= rem, so a multi-candidate round can at most reach
    the budget, never cross it. The candidate that crosses the budget is therefore
    always the only one drawn for its prompt in that round, and nothing is drawn past
    it: spent <= budget + max_new_tokens.

    Batching does not suffer, because the batch is wide across PROMPTS: 500 active
    prompts at k=1 is already a 500-request vLLM call. Per-prompt fan-out only matters
    at the tail, where few prompts remain but their remaining budgets are large — which
    is exactly where floor(rem / max_new_tokens) is itself large.
    """
    want = []
    for rem, n in zip(remaining, n_drawn):
        if rem <= 0 or n >= max_n:
            want.append(0)
        else:
            k = int(rem // max(max_new_tokens, 1))
            want.append(max(1, min(k, per_prompt_cap, max_n - n)))

    total = sum(want)
    if total > max_concurrent:
        scale = max_concurrent / total
        want = [max(1, int(w * scale)) if w else 0 for w in want]
    return want


def summarize_budget_realization(
    realized, budgets, n_candidates, max_new_tokens, max_n, tolerance=0.10
):
    """Check and describe how closely BoN matched its token budget.

    `realized` is the decode-token count of all candidates BoN drew for a problem.
    Because the candidate that crosses the budget is kept, a problem normally ends at
    budget <= realized <= budget + max_new_tokens; it can end below budget only by
    hitting max_n. Expected overshoot is about half a candidate per problem.

    Returns a dict of statistics and a list of warnings. Deliberately never raises:
    aborting after hours of GPU time is worse than reporting loudly and saving the CSV.
    """
    realized = np.asarray(realized, dtype=np.int64)
    budgets = np.asarray(budgets, dtype=np.int64)
    n_candidates = np.asarray(n_candidates, dtype=np.int64)

    total_ratio = float(realized.sum()) / max(int(budgets.sum()), 1)
    ratios = realized / np.maximum(budgets, 1)
    capped = n_candidates >= max_n

    warnings = []
    if total_ratio > 1.0 + tolerance:
        warnings.append(
            f"aggregate token ratio {total_ratio:.3f} is above {1 + tolerance:.2f} — "
            f"BoN was given materially more compute than mcmc_cut"
        )
    short = np.flatnonzero((realized < budgets) & ~capped)
    if short.size:
        warnings.append(
            f"{short.size} problem(s) stopped below budget without hitting max_n, which "
            f"the stopping rule should make impossible (first: index {int(short[0])}, "
            f"{int(realized[short[0]])} < {int(budgets[short[0]])})"
        )
    over = np.flatnonzero(realized > budgets + max_new_tokens)
    if over.size:
        warnings.append(
            f"{over.size} problem(s) overshot budget by more than one candidate, which "
            f"round sizing should make impossible (first: index {int(over[0])}, "
            f"{int(realized[over[0]])} > {int(budgets[over[0]])} + {max_new_tokens})"
        )

    stats = {
        "total_ratio": round(total_ratio, 4),
        "ratio_median": round(float(np.median(ratios)), 4),
        "ratio_p95": round(float(np.percentile(ratios, 95)), 4),
        "n_median": int(np.median(n_candidates)),
        "n_min": int(n_candidates.min()),
        "n_max": int(n_candidates.max()),
        "n_eq_1": int((n_candidates == 1).sum()),
        "n_at_max_n": int(capped.sum()),
        "warnings": warnings,
    }
    return stats


# ------------------------------------------------------------------------- selftest


def selftest():
    failures = []

    def check(name, condition, detail=""):
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not condition:
            failures.append(name)

    MNT = 3072

    print("1. plan_round_sizes")
    k = plan_round_sizes([100000], [0], max_n=256, max_new_tokens=MNT, per_prompt_cap=16)
    check("per_prompt_cap binds", k == [16], f"got {k}")
    k = plan_round_sizes([100000], [0], max_n=256, max_new_tokens=MNT, per_prompt_cap=64)
    check("worst-case rule: 100000//3072", k == [32], f"got {k}")
    k = plan_round_sizes([6336], [0], max_n=256, max_new_tokens=MNT)
    check("6336 budget -> 2", k == [2], f"got {k}")
    k = plan_round_sizes([400], [0], max_n=256, max_new_tokens=MNT)
    check("rem << max_new -> 1 (never 0 while active)", k == [1], f"got {k}")
    k = plan_round_sizes([5000], [256], max_n=256, max_new_tokens=MNT)
    check("n_drawn == max_n -> 0", k == [0], f"got {k}")
    k = plan_round_sizes([0], [3], max_n=256, max_new_tokens=MNT)
    check("rem <= 0 -> 0", k == [0], f"got {k}")
    k = plan_round_sizes([50000], [255], max_n=256, max_new_tokens=MNT)
    check("max_n - n_drawn binds", k == [1], f"got {k}")
    check("a multi-candidate round can never cross the budget",
          all(ki * MNT <= rem
              for rem in (3072, 6336, 10000, 99999)
              for ki in plan_round_sizes([rem], [0], max_n=256, max_new_tokens=MNT,
                                         per_prompt_cap=1000)))
    k = plan_round_sizes([100000] * 200, [0] * 200,
                         max_n=256, max_new_tokens=MNT, per_prompt_cap=16,
                         max_concurrent=1024)
    check("max_concurrent scales down", sum(k) <= 1024, f"sum={sum(k)}")
    check("max_concurrent keeps every active prompt >= 1", min(k) >= 1, f"min={min(k)}")

    print("2. stopping rule under simulation: the crossing candidate is kept")
    # Mirrors _batch_best_of_n: draw until spent >= budget, keeping the crossing draw.
    rng = np.random.default_rng(0)
    max_new_tokens, max_n, n_probs = 3072, 256, 500
    budgets = np.maximum(
        (rng.lognormal(np.log(6336), 0.8, n_probs)).astype(int), 2000
    )
    spent = np.zeros(n_probs, dtype=np.int64)
    drawn = np.zeros(n_probs, dtype=np.int64)
    ignored = 0
    active = list(range(n_probs))
    rounds = 0
    while active:
        rounds += 1
        k = plan_round_sizes(
            [int(budgets[i] - spent[i]) for i in active],
            [int(drawn[i]) for i in active],
            max_n=max_n, max_new_tokens=max_new_tokens,
            per_prompt_cap=16, max_concurrent=4096,
        )
        req = [i for i, ki in zip(active, k) for _ in range(ki)]
        if not req:
            break
        for i in req:
            length = max(int(min(rng.lognormal(np.log(428), 0.7), max_new_tokens)), 1)
            if spent[i] >= budgets[i]:
                ignored += 1
                continue
            spent[i] += length
            drawn[i] += 1
        active = [i for i in active if spent[i] < budgets[i] and drawn[i] < max_n]

    check("every problem reaches its budget", bool((spent >= budgets).all()),
          f"{int((spent < budgets).sum())} short")
    check("overshoot is at most one candidate",
          bool((spent <= budgets + max_new_tokens).all()))
    check("no candidate is drawn past the budget", ignored == 0, f"{ignored} ignored")
    ratio = spent.sum() / budgets.sum()
    check("aggregate ratio within 10% above budget", 1.0 <= ratio <= 1.10,
          f"ratio={ratio:.4f}")
    print(f"        rounds={rounds}  ratio={ratio:.4f}  "
          f"N median={int(np.median(drawn))} min={int(drawn.min())} max={int(drawn.max())}")

    print("3. summarize_budget_realization")
    s = summarize_budget_realization(
        [6500, 6400], [6336, 6336], [14, 15], max_new_tokens=3072, max_n=256
    )
    check("clean case has no warnings", s["warnings"] == [], f"{s['warnings']}")
    s = summarize_budget_realization(
        [100, 100], [6336, 6336], [3, 3], max_new_tokens=3072, max_n=256
    )
    check("stopping short is flagged", any("below budget" in w for w in s["warnings"]))
    s = summarize_budget_realization(
        [100], [6336], [256], max_new_tokens=3072, max_n=256
    )
    check("stopping short at max_n is allowed", s["warnings"] == [], f"{s['warnings']}")
    s = summarize_budget_realization(
        [20000], [6336], [5], max_new_tokens=3072, max_n=256
    )
    check("overshoot beyond one candidate is flagged",
          any("overshot" in w for w in s["warnings"]))
    check("gross over-use is flagged", any("above" in w for w in s["warnings"]))

    print("\n" + ("SELFTEST FAILED: " + ", ".join(failures) if failures else "selftest ok"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(selftest())
