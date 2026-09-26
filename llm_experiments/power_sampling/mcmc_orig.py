"""Blockwise MCMC power sampling with uniform cut points, reproducing the original
reasoning-with-sampling Metropolis-Hastings ratio (the `mcmc_orig` arm).
"""

import gc
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from .outputs import McmcHistory, McmcOutput, McmcStats, NaiveTempOutput


class McmcOrigSampler:
    """Blockwise MH sampler targeting p^(1/temp) with uniformly drawn cut points."""

    def __init__(self, base):
        self.base = base

    def batch_mcmc_orig_power_sample(
        self,
        batch_prompt_ids,
        temp,
        mcmc_steps,
        max_new_tokens,
        num_blocks=16,
        top_k=50,
        store_history=False,
        batch_size=256,
    ) -> list[McmcOutput]:
        """Blockwise MCMC sampler matching the original paper for a batch of prompts, processed in mini-batches.

        Wraps `_batch_mcmc_orig_power_sample` and processes `batch_prompt_ids` in
        chunks of `batch_size`, freeing GPU memory between chunks.

        Args:
            batch_prompt_ids: List[List[int]] — prompt token ids for each problem.
            temp: Temperature = 1/alpha. Lower temp => sharper target.
            mcmc_steps: Number of MCMC refinement steps per block.
            max_new_tokens: Total new tokens to generate across all blocks.
            num_blocks: Number of blocks to split generation into (default 16).
            top_k: Top-k filtering for vLLM generation.
            store_history: Whether to store MCMC history.
            batch_size: Mini-batch size for processing prompts (default 256).

        Returns:
            List[McmcOutput] — one per prompt, in the same order as `batch_prompt_ids`.
        """
        all_outputs = []
        for i in tqdm(
            range(0, len(batch_prompt_ids), batch_size),
            desc="MCMC orig mini-batches",
            leave=False,
        ):
            chunk_prompt_ids = batch_prompt_ids[i : i + batch_size]
            chunk_outputs = self._batch_mcmc_orig_power_sample(
                chunk_prompt_ids,
                temp,
                mcmc_steps,
                max_new_tokens,
                num_blocks,
                top_k,
                store_history,
            )
            all_outputs.extend(chunk_outputs)

            gc.collect()
            torch.cuda.empty_cache()

        return all_outputs

    def _batch_mcmc_orig_power_sample(
        self,
        batch_prompt_ids,
        temp,
        mcmc_steps,
        max_new_tokens,
        num_blocks=16,
        top_k=50,
        store_history=False,
    ):
        """Blockwise MCMC sampler matching the original paper for a batch of prompts.

        Reproduces the MH ratio of the original reasoning-with-sampling implementation:
        - Truncates both current and proposal suffixes to min(len_cur, len_prop)
        - No cut-point correction term

        Args:
            batch_prompt_ids: List[List[int]] — prompt token ids for each
                problem in the batch.
            temp: Temperature = 1/alpha. Lower temp => sharper target.
            mcmc_steps: Number of MCMC refinement steps per block.
            max_new_tokens: Total new tokens to generate across all blocks.
            num_blocks: Number of blocks to split generation into (default 16).

        Returns:
            List[McmcOutput] — one per prompt, in the same order as
            `batch_prompt_ids`.
        """
        _t0 = time.monotonic()
        assert (
            max_new_tokens % num_blocks == 0
        ), f"_batch_mcmc_orig_power_sample: {max_new_tokens} % {num_blocks} != 0"
        jump_size = max_new_tokens // num_blocks

        batch_size = len(batch_prompt_ids)

        # Per-prompt MCMC state
        all_output_ids = [[] for _ in range(batch_size)]
        all_logprobs = [[] for _ in range(batch_size)]
        all_raw_logprobs = [[] for _ in range(batch_size)]
        all_raw_entropies = [[] for _ in range(batch_size)]
        all_stats = [McmcStats() for _ in range(batch_size)]
        all_histories = [
            McmcHistory() if store_history else None for _ in range(batch_size)
        ]
        active = list(range(batch_size))

        for block_idx in tqdm(range(num_blocks), desc="Generating blocks", leave=False):
            # --- Step 1: Extend each unfinished sequence by jump_size tokens ---
            if not active:
                break

            if store_history:
                for i in active:
                    all_histories[i].new_block()

            init_results = self.base.batch_naive_temp(
                [batch_prompt_ids[i] + all_output_ids[i] for i in active],
                temp=temp,
                sample_len=jump_size,
                ignore_eos=False,
                top_k=top_k,
            )

            for idx, init_output in zip(active, init_results):
                all_output_ids[idx].extend(init_output.output_ids)
                all_logprobs[idx].extend(init_output.logprobs)
                all_raw_logprobs[idx].extend(init_output.raw_logprobs)
                all_raw_entropies[idx].extend(init_output.raw_entropies)

            # --- Step 2: MCMC refinement via Metropolis-Hastings ---
            block_attempts = {i: 0 for i in active}
            block_acceptances = {i: 0 for i in active}

            for _step in tqdm(range(mcmc_steps), desc="MCMC steps", leave=False):

                if not active:
                    break

                # Pick a random cut-point for each active prompt
                suffix_starts = {}
                for i in active:
                    block_attempts[i] += 1
                    t = len(all_output_ids[i])
                    suffix_starts[i] = random.randint(0, t - 1)

                # Propose: regenerate from cut-point to current total length
                prop_contexts = [
                    batch_prompt_ids[i] + all_output_ids[i][: suffix_starts[i]]
                    for i in active
                ]
                prop_sample_lens = [
                    len(all_output_ids[i]) - suffix_starts[i] for i in active
                ]

                prop_results = self.base.batch_naive_temp(
                    prop_contexts,
                    temp=temp,
                    sample_len=prop_sample_lens,
                    min_len=1,
                    ignore_eos=False,
                    top_k=top_k,
                )

                # Apply MH acceptance for each active prompt
                for i, prop_output in zip(active, prop_results):
                    ss = suffix_starts[i]

                    # Truncate to min length (matching original paper)
                    len_prop = len(prop_output.logprobs)
                    suffix_logprobs = all_logprobs[i][ss : ss + len_prop]
                    suffix_raw_logprobs = all_raw_logprobs[i][ss : ss + len_prop]

                    log_acceptance_ratio = (
                        np.sum(prop_output.raw_logprobs) / temp
                        - np.sum(suffix_raw_logprobs) / temp
                        + np.sum(suffix_logprobs)
                        - np.sum(prop_output.logprobs)
                    )

                    acceptance_prob = min(1.0, float(np.exp(log_acceptance_ratio)))
                    accepted = np.random.rand() < acceptance_prob

                    if all_histories[i] is not None:
                        t = len(all_output_ids[i])
                        current = NaiveTempOutput(
                            output_ids=list(all_output_ids[i]),
                            logprobs=list(all_logprobs[i]),
                            raw_logprobs=list(all_raw_logprobs[i]),
                            raw_entropies=list(all_raw_entropies[i]),
                            elapsed_time=0.0,
                        )
                        all_histories[i].record_step(
                            current=current,
                            cut_index=ss,
                            cut_probs=np.ones(t) / t,
                            proposal=prop_output,
                            acceptance_prob=acceptance_prob,
                            accepted=bool(accepted),
                        )

                    if accepted:
                        block_acceptances[i] += 1
                        all_output_ids[i][ss:] = prop_output.output_ids
                        all_logprobs[i][ss:] = prop_output.logprobs
                        all_raw_logprobs[i][ss:] = prop_output.raw_logprobs
                        all_raw_entropies[i][ss:] = prop_output.raw_entropies

                        assert (
                            len(all_output_ids[i])
                            == len(all_logprobs[i])
                            == len(all_raw_logprobs[i])
                            == len(all_raw_entropies[i])
                        )

            # Record stats and check for EOS
            new_active = []
            for i in active:
                all_stats[i].attempts_per_block.append(block_attempts[i])
                all_stats[i].accepts_per_block.append(block_acceptances[i])

                if self.base.tokenizer.eos_token_id in all_output_ids[i]:
                    eos_idx = all_output_ids[i].index(self.base.tokenizer.eos_token_id)
                    all_output_ids[i] = all_output_ids[i][: eos_idx + 1]
                    all_logprobs[i] = all_logprobs[i][: eos_idx + 1]
                    all_raw_logprobs[i] = all_raw_logprobs[i][: eos_idx + 1]
                    all_raw_entropies[i] = all_raw_entropies[i][: eos_idx + 1]
                else:
                    new_active.append(i)
            active = new_active

        elapsed = (time.monotonic() - _t0) / batch_size if batch_size > 0 else 0.0
        return [
            McmcOutput(
                all_output_ids[i],
                all_logprobs[i],
                all_raw_logprobs[i],
                all_raw_entropies[i],
                all_stats[i],
                elapsed_time=elapsed,
                history=all_histories[i],
            )
            for i in range(batch_size)
        ]
