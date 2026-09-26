"""Compute-matched Best-of-N sampling (the `bon_std` / `bon_naive` arms)."""

import gc
import time

import numpy as np
import torch
from tqdm import tqdm

from .bon_utils import BON_MAX_N, plan_round_sizes
from .outputs import BestOfNOutput


class BestOfNSampler:
    """Best-of-N under a per-prompt decode-token budget, selecting by mean base-model log-likelihood."""

    def __init__(self, base):
        self.base = base

    def batch_best_of_n(
        self,
        batch_prompt_ids,
        temp,
        max_new_tokens,
        token_budgets,
        top_k=50,
        max_n=BON_MAX_N,
        per_prompt_cap=16,
        max_concurrent=1024,
        batch_size=256,
    ) -> list[BestOfNOutput]:
        """Best-of-N under a per-prompt decode-token budget, selected by mean logprob.

        Draws independent candidates for each prompt until their total number of
        decode tokens reaches its budget, rounded up: the candidate that crosses the
        budget is kept. Returns the candidate with the highest average base-model
        log-likelihood.

        Args:
            batch_prompt_ids: List[List[int]] — tokenized prompts.
            temp: Sampling temperature for candidates (1.0 = standard sampling).
            max_new_tokens: Per-candidate generation cap.
            token_budgets: List[int] — decode-token budget per prompt, one-to-one with
                batch_prompt_ids: McmcOutput.n_generated_tokens of the matched
                entropy-cut run, rejected proposals included.
            top_k: Top-k truncation; must match the arm being compared against so both
                score log-probabilities under the same truncated model.
            max_n: Hard cap on candidates per prompt.
            per_prompt_cap: Cap on candidates drawn for one prompt in a single round.
            max_concurrent: Cap on total in-flight requests per round.
            batch_size: Prompts per mini-batch. Matches batch_mcmc_power_sample so the
                amortised elapsed_time values are comparable.

        Returns:
            List of BestOfNOutput, one per prompt, in input order.
        """
        assert len(token_budgets) == len(batch_prompt_ids), (
            f"batch_best_of_n: len(token_budgets)={len(token_budgets)} "
            f"!= len(batch_prompt_ids)={len(batch_prompt_ids)}"
        )
        assert all(b > 0 for b in token_budgets), "batch_best_of_n: budget must be > 0"

        all_outputs = []
        for i in tqdm(
            range(0, len(batch_prompt_ids), batch_size),
            desc="BoN mini-batches",
            leave=False,
        ):
            chunk_prompt_ids = batch_prompt_ids[i : i + batch_size]
            chunk_budgets = token_budgets[i : i + batch_size]
            chunk_outputs = self._batch_best_of_n(
                chunk_prompt_ids,
                temp,
                max_new_tokens,
                chunk_budgets,
                top_k,
                max_n,
                per_prompt_cap,
                max_concurrent,
            )
            all_outputs.extend(chunk_outputs)
            gc.collect()
            torch.cuda.empty_cache()

        return all_outputs

    def _batch_best_of_n(
        self,
        batch_prompt_ids,
        temp,
        max_new_tokens,
        token_budgets,
        top_k=50,
        max_n=BON_MAX_N,
        per_prompt_cap=16,
        max_concurrent=1024,
    ) -> list[BestOfNOutput]:
        """Single mini-batch of batch_best_of_n. See that method for semantics."""
        _t0 = time.monotonic()
        batch_size = len(batch_prompt_ids)

        spent_tokens = [0] * batch_size
        n_drawn = [0] * batch_size
        best = [None] * batch_size
        best_score = [-np.inf] * batch_size
        stop_reason = ["budget"] * batch_size
        active = list(range(batch_size))

        while active:
            round_sizes = plan_round_sizes(
                [token_budgets[i] - spent_tokens[i] for i in active],
                [n_drawn[i] for i in active],
                max_n=max_n,
                max_new_tokens=max_new_tokens,
                per_prompt_cap=per_prompt_cap,
                max_concurrent=max_concurrent,
            )
            req_idx = [i for i, k in zip(active, round_sizes) for _ in range(k)]
            if not req_idx:
                break

            # seed stays None: batch_naive_temp applies one scalar seed to every
            # request, so a shared seed over duplicated prompts would return N
            # byte-identical candidates.
            outs = self.base.batch_naive_temp(
                [batch_prompt_ids[i] for i in req_idx],
                temp=temp,
                sample_len=max_new_tokens,
                min_len=0,
                ignore_eos=False,
                seed=None,
                top_k=top_k,
                use_tqdm=False,
            )

            for i, out in zip(req_idx, outs):
                # plan_round_sizes keeps a multi-candidate round within the remaining
                # budget, so only the last candidate of a prompt's round can reach it;
                # this guard only matters if that invariant is ever broken.
                if spent_tokens[i] >= token_budgets[i]:
                    continue
                spent_tokens[i] += out.n_generated_tokens
                n_drawn[i] += 1
                score = (
                    float(np.mean(out.raw_logprobs)) if out.raw_logprobs else -np.inf
                )
                if best[i] is None or score > best_score[i]:
                    best_score[i] = score
                    best[i] = out

            new_active = []
            for i in active:
                if spent_tokens[i] >= token_budgets[i]:
                    continue  # budget reached; the crossing candidate was kept
                if n_drawn[i] >= max_n:
                    stop_reason[i] = "max_n"
                    continue
                new_active.append(i)
            active = new_active

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = (time.monotonic() - _t0) / batch_size if batch_size > 0 else 0.0

        return [
            BestOfNOutput(
                best[i].output_ids,
                best[i].logprobs,
                best[i].raw_logprobs,
                best[i].raw_entropies,
                avg_logprob=best_score[i],
                n_candidates=n_drawn[i],
                n_generated_tokens=spent_tokens[i],
                token_budget=token_budgets[i],
                stop_reason=stop_reason[i],
                elapsed_time=elapsed,
            )
            for i in range(batch_size)
        ]
