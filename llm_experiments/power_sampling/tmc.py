"""Tree/rollout Monte Carlo power sampling (the `tmc` arm)."""

import gc
import time

import numpy as np
import scipy.special
import torch
from tqdm import tqdm

from .outputs import McmcOutput, McmcStats
from .utils import unflatten_list


class TmcSampler:
    """Blockwise sampler that scores top candidates with short rollouts (arXiv 2601.21590, Alg. 2)."""

    def __init__(self, base):
        self.base = base

    def batch_tmc_power_sample(
        self,
        batch_prompt_ids,
        temp,
        max_new_tokens,
        block_size=192,
        num_candidates=64,
        num_top_candidates=8,
        num_rollouts=8,
        horizon_len=192,
        top_k=50,
        batch_size=4,
    ) -> list[McmcOutput]:
        """
        Run TMC Power Sampling for a batch of prompts, processing them in mini-batches.

        Args:
            batch_prompt_ids: List of List of tokenized prompt IDs.
            temp: Temperature = 1/alpha. Lower temp => sharper target.
            max_new_tokens: Total new tokens to generate.
            block_size: Size of each generation chunk.
            num_candidates: Number of raw candidate blocks generated from base model.
            num_top_candidates: Top-K candidates retained for rollout evaluation.
            num_rollouts: Number of independent lookahead rollouts per candidate.
            horizon_len: Length of the lookahead rollouts.
            top_k: Top-k filtering for the base vLLM generation.
            batch_size: Mini-batch size for processing prompts.

        Returns:
            List[McmcOutput], one per prompt in `batch_prompt_ids`.
        """
        all_outputs = []
        for i in tqdm(
            range(0, len(batch_prompt_ids), batch_size),
            desc="TMC mini-batches",
            leave=False,
        ):
            chunk_prompt_ids = batch_prompt_ids[i : i + batch_size]
            chunk_outputs = self._batch_tmc_power_sample(
                chunk_prompt_ids,
                temp,
                max_new_tokens,
                block_size,
                num_candidates,
                num_top_candidates,
                num_rollouts,
                horizon_len,
                top_k,
            )
            all_outputs.extend(chunk_outputs)

            gc.collect()
            torch.cuda.empty_cache()

        return all_outputs

    def _batch_tmc_power_sample(
        self,
        batch_prompt_ids,
        temp,
        max_new_tokens,
        block_size=192,
        num_candidates=64,
        num_top_candidates=8,
        num_rollouts=8,
        horizon_len=192,
        top_k=50,
    ):
        """
        Batched Twisted Monte Carlo (TMC) Power Sampling.
        Processes multiple prompts simultaneously, flattening candidate generation
        and rollout steps into single batch_naive_temp calls for GPU efficiency.

        Args:
            batch_prompt_ids: List of List of tokenized prompt IDs.
            temp: Temperature = 1/alpha. Lower temp => sharper target.
            max_new_tokens: Total new tokens to generate.
            block_size: Size of each generation chunk.
            num_candidates: Number of raw candidate blocks generated from base model.
            num_top_candidates: Top-K candidates retained for rollout evaluation.
            num_rollouts: Number of independent lookahead rollouts per candidate.
            horizon_len: Length of the lookahead rollouts.
            top_k: Top-k filtering for the base vLLM generation.

        Returns:
            List[McmcOutput], one per prompt in `batch_prompt_ids`.
        """
        assert (
            max_new_tokens % block_size == 0
        ), f"_batch_tmc_power_sample: {max_new_tokens} % {block_size} != 0"

        _t0 = time.monotonic()
        num_blocks = max_new_tokens // block_size
        alpha = 1.0 / temp
        batch_size = len(batch_prompt_ids)

        # Per-prompt accumulators
        all_output_ids = [[] for _ in range(batch_size)]
        all_logprobs = [[] for _ in range(batch_size)]
        all_raw_logprobs = [[] for _ in range(batch_size)]
        all_raw_entropies = [[] for _ in range(batch_size)]
        all_stats = [McmcStats(temp=temp, top_k=top_k) for _ in range(batch_size)]

        # Track which prompts are still active (haven't hit EOS)
        active_mask = [True] * batch_size

        for block_idx in tqdm(range(num_blocks), desc="TMC blocks", leave=False):
            active_indices = [i for i in range(batch_size) if active_mask[i]]
            if not active_indices:
                break

            # --- Step 1: Generate num_candidates candidate blocks for each active prompt ---
            candidate_contexts = []
            for i in active_indices:
                current_context = batch_prompt_ids[i] + all_output_ids[i]
                candidate_contexts.extend([current_context] * num_candidates)

            candidate_outputs_flat = self.base.batch_naive_temp(
                candidate_contexts,
                temp=temp,
                sample_len=block_size,
                ignore_eos=False,
                top_k=top_k,
                use_tqdm=True,
            )

            # Reshape: (num_active, num_candidates)
            candidate_outputs = unflatten_list(
                candidate_outputs_flat,
                [num_candidates] * len(active_indices),
            )

            # --- Step 2: Keep top num_top_candidates per prompt ---
            top_candidates_per_prompt = []
            for idx, i in enumerate(active_indices):
                candidate_out = candidate_outputs[idx]
                candidate_scores = [np.sum(out.raw_logprobs) for out in candidate_out]
                top_k_indices = np.argpartition(candidate_scores, -num_top_candidates)[
                    -num_top_candidates:
                ]
                top_candidates_per_prompt.append(
                    [candidate_out[j] for j in top_k_indices]
                )

            # --- Step 3: Monte Carlo Rollouts for each top candidate ---
            # Compute per-prompt active top candidate indices (those without EOS)
            active_top_indices_per_prompt = [
                [
                    c
                    for c, candidate in enumerate(top_candidates_per_prompt[idx])
                    if self.base.tokenizer.eos_token_id not in candidate.output_ids
                ]
                for idx in range(len(active_indices))
            ]

            rollout_contexts = []
            rollout_counts = []
            # number of rollout contexts contributed per active prompt
            for idx, i in enumerate(active_indices):
                current_context = batch_prompt_ids[i] + all_output_ids[i]
                active_top_indices = active_top_indices_per_prompt[idx]
                for active_c in active_top_indices:
                    candidate = top_candidates_per_prompt[idx][active_c]
                    cand_context = current_context + candidate.output_ids
                    rollout_contexts.extend([cand_context] * num_rollouts)
                rollout_counts.append(len(active_top_indices) * num_rollouts)

            rollouts_outputs_flat = []
            if rollout_contexts:
                rollouts_outputs_flat = self.base.batch_naive_temp(
                    rollout_contexts,
                    temp=temp,
                    sample_len=horizon_len,
                    ignore_eos=False,
                    top_k=top_k,
                    use_tqdm=True,
                )

            # Reshape: per active prompt, a flat list of rollouts for active top candidates only
            rollouts_per_prompt_flat = unflatten_list(
                rollouts_outputs_flat, rollout_counts
            )

            # --- Step 4 & 5: Compute twisted probs with jackknife + sample ---
            for idx, i in enumerate(active_indices):
                top_candidates = top_candidates_per_prompt[idx]
                active_top_indices = active_top_indices_per_prompt[idx]
                rollouts_flat_i = rollouts_per_prompt_flat[idx]

                # Group rollouts: (num_top_candidates, num_rollouts), [] for EOS candidates
                rollouts = [[] for _ in range(num_top_candidates)]
                for j, active_c in enumerate(active_top_indices):
                    rollouts[active_c] = rollouts_flat_i[
                        j * num_rollouts : (j + 1) * num_rollouts
                    ]

                # Compute Jackknife Corrected Scaling Factors
                log_twist_hat = np.zeros(num_top_candidates)
                log_twist_hat_loo = np.zeros((num_top_candidates, num_rollouts))

                for active_c in active_top_indices:
                    log_rollout_probs_power = np.array(
                        [
                            alpha * np.sum(r.raw_logprobs) - np.sum(r.logprobs)
                            for r in rollouts[active_c]
                        ]
                    )
                    log_twist_hat[active_c] = scipy.special.logsumexp(
                        log_rollout_probs_power
                    ) - np.log(num_rollouts)

                    if num_rollouts > 1:
                        for s in range(num_rollouts):
                            S_loo = np.delete(log_rollout_probs_power, s)
                            log_twist_hat_loo[active_c, s] = scipy.special.logsumexp(
                                S_loo
                            ) - np.log(num_rollouts - 1)
                    else:
                        log_twist_hat_loo[active_c, 0] = log_twist_hat[active_c]

                # Full standard power probabilities
                log_twisted_candidate_probs = np.array(
                    [
                        alpha * np.sum(top_candidates[c].raw_logprobs)
                        + log_twist_hat[c]
                        for c in range(num_top_candidates)
                    ]
                )
                twisted_candidate_probs = scipy.special.softmax(
                    log_twisted_candidate_probs
                )

                # Leave-One-Out & Jackknife correction
                if num_rollouts > 1:
                    twisted_candidate_probs_loo = np.zeros(
                        (num_top_candidates, num_rollouts)
                    )
                    for rollout_s in range(num_rollouts):
                        log_twisted_candidate_probs_loo = np.array(
                            [
                                alpha * np.sum(top_candidates[c].raw_logprobs)
                                + log_twist_hat_loo[c, rollout_s]
                                for c in range(num_top_candidates)
                            ]
                        )
                        twisted_candidate_probs_loo[:, rollout_s] = (
                            scipy.special.softmax(log_twisted_candidate_probs_loo)
                        )

                    twisted_candidate_probs_jackknife = (
                        num_rollouts * twisted_candidate_probs
                        - ((num_rollouts - 1.0) / num_rollouts)
                        * np.sum(twisted_candidate_probs_loo, axis=1)
                    )
                else:
                    twisted_candidate_probs_jackknife = twisted_candidate_probs

                # Ensure valid probability distribution
                twisted_candidate_probs_jackknife = np.clip(
                    twisted_candidate_probs_jackknife, 0, None
                )
                sum_p = np.sum(twisted_candidate_probs_jackknife)
                if sum_p > 0:
                    twisted_candidate_probs_jackknife /= sum_p
                else:
                    twisted_candidate_probs_jackknife = (
                        np.ones(num_top_candidates) / num_top_candidates
                    )

                # Sample the block
                sampled_idx = np.random.choice(
                    num_top_candidates, p=twisted_candidate_probs_jackknife
                )
                sampled_candidate = top_candidates[sampled_idx]

                all_output_ids[i].extend(sampled_candidate.output_ids)
                all_logprobs[i].extend(sampled_candidate.logprobs)
                all_raw_logprobs[i].extend(sampled_candidate.raw_logprobs)
                all_raw_entropies[i].extend(sampled_candidate.raw_entropies)

                all_stats[i].attempts_per_block.append(
                    num_candidates + (num_top_candidates * num_rollouts)
                )
                all_stats[i].accepts_per_block.append(1)

                # Early termination on EOS
                if self.base.tokenizer.eos_token_id in sampled_candidate.output_ids:
                    eos_idx = all_output_ids[i].index(self.base.tokenizer.eos_token_id)
                    all_output_ids[i] = all_output_ids[i][: eos_idx + 1]
                    all_logprobs[i] = all_logprobs[i][: eos_idx + 1]
                    all_raw_logprobs[i] = all_raw_logprobs[i][: eos_idx + 1]
                    all_raw_entropies[i] = all_raw_entropies[i][: eos_idx + 1]
                    active_mask[i] = False

        elapsed = (time.monotonic() - _t0) / batch_size if batch_size > 0 else 0.0
        return [
            McmcOutput(
                all_output_ids[i],
                all_logprobs[i],
                all_raw_logprobs[i],
                all_raw_entropies[i],
                all_stats[i],
                elapsed_time=elapsed,
            )
            for i in range(batch_size)
        ]
