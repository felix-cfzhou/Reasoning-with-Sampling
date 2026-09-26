"""Entropy-guided blockwise MCMC power sampling (the `mcmc_cut` arm)."""

import gc
import time

import numpy as np
import torch
from tqdm import tqdm

from .outputs import McmcHistory, McmcOutput, McmcStats, NaiveTempOutput


class McmcCutSampler:
    """Blockwise MH sampler targeting p^(1/temp) with entropy-increase-weighted cut points."""

    def __init__(self, base):
        self.base = base

    def batch_mcmc_power_sample(
        self,
        batch_prompt_ids,
        temp,
        mcmc_steps,
        max_new_tokens,
        num_blocks=16,
        top_k=50,
        cut_power=4.0,
        store_history=False,
        batch_size=256,
    ) -> list[McmcOutput]:
        """Blockwise MCMC sampler targeting p^alpha for a batch of prompts, processed in mini-batches.

        Wraps `_batch_mcmc_power_sample` and processes `batch_prompt_ids` in
        chunks of `batch_size`, freeing GPU memory between chunks.

        Args:
            batch_prompt_ids: List[List[int]] — prompt token ids for each problem.
            temp: Temperature = 1/alpha. Lower temp => sharper target.
            mcmc_steps: Number of MCMC refinement steps per block.
            max_new_tokens: Total new tokens to generate across all blocks.
            num_blocks: Number of blocks to split generation into (default 16).
            top_k: Top-k filtering for vLLM generation.
            cut_power: Cut-point entropy weighting exponent (beta).
            store_history: Whether to store MCMC history.
            batch_size: Mini-batch size for processing prompts (default 256).

        Returns:
            List[McmcOutput] — one per prompt, in the same order as `batch_prompt_ids`.
        """
        all_outputs = []
        for i in tqdm(
            range(0, len(batch_prompt_ids), batch_size),
            desc="MCMC mini-batches",
            leave=False,
        ):
            chunk_prompt_ids = batch_prompt_ids[i : i + batch_size]
            chunk_outputs = self._batch_mcmc_power_sample(
                chunk_prompt_ids,
                temp,
                mcmc_steps,
                max_new_tokens,
                num_blocks,
                top_k,
                cut_power,
                store_history,
            )
            all_outputs.extend(chunk_outputs)

            gc.collect()
            torch.cuda.empty_cache()

        return all_outputs

    def _batch_mcmc_power_sample(
        self,
        batch_prompt_ids,
        temp,
        mcmc_steps,
        max_new_tokens,
        num_blocks=16,
        top_k=50,
        cut_power=4.0,
        store_history=False,
    ):
        """Blockwise MCMC sampler targeting p^alpha for a batch of prompts.

        All prompts advance in lockstep: each block extension and each MCMC
        proposal is a single `batch_naive_temp` call across the active prompts.

        Args:
            batch_prompt_ids: List[List[int]] — prompt token ids for each
                problem in the batch.
            temp: Temperature = 1/alpha. Lower temp => sharper target.
            mcmc_steps: Number of MCMC refinement steps per block.
            max_new_tokens: Total new tokens to generate across all blocks.
            num_blocks: Number of blocks to split generation into (default 16).

        Returns:
            List[McmcOutput] — one per prompt, in the same order as
            `batch_prompt_ids`. Each McmcOutput contains:
                - output_ids: List[int] — final token sequence (generated tokens).
                - logprobs: List[float] — proposal log-probs for each generated token.
                - raw_logprobs: List[float] — base-model log-probs for each generated token.
                - raw_entropies: List[float] — base-model entropy at each generated token.
                - stats: McmcStats — acceptance/attempt counts per block.
        """
        assert (
            max_new_tokens % num_blocks == 0
        ), f"_batch_mcmc_power_sample: {max_new_tokens} % {num_blocks} != 0"
        jump_size = max_new_tokens // num_blocks

        _t0 = time.monotonic()
        batch_size = len(batch_prompt_ids)

        # Per-prompt MCMC state
        all_output_ids = [[] for _ in range(batch_size)]
        all_logprobs = [[] for _ in range(batch_size)]
        all_raw_logprobs = [[] for _ in range(batch_size)]
        all_raw_entropies = [[] for _ in range(batch_size)]
        all_stats = [
            McmcStats(temp=temp, top_k=top_k, cut_power=cut_power)
            for _ in range(batch_size)
        ]
        all_histories = [
            McmcHistory() if store_history else None for _ in range(batch_size)
        ]
        # Compute accounting: every token the engine decodes for this prompt, whether or
        # not it survives into the final chain. This is the compute-matched Best-of-N budget.
        all_gen_tokens = [0] * batch_size
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
                sample_len=[jump_size for i in active],
                ignore_eos=False,
                top_k=top_k,
            )

            for idx, init_output in zip(active, init_results):
                all_output_ids[idx].extend(init_output.output_ids)
                all_logprobs[idx].extend(init_output.logprobs)
                all_raw_logprobs[idx].extend(init_output.raw_logprobs)
                all_raw_entropies[idx].extend(init_output.raw_entropies)
                all_gen_tokens[idx] += init_output.n_generated_tokens

            # --- Step 2: MCMC refinement via Metropolis-Hastings ---
            block_attempts = {i: 0 for i in active}
            block_acceptances = {i: 0 for i in active}
            block_power = {i: [] for i in active}
            block_proposal = {i: [] for i in active}
            block_cut = {i: [] for i in active}

            for _step in tqdm(range(mcmc_steps), desc="MCMC steps", leave=False):

                if not active:
                    break

                # Pick a random cut-point for each active prompt (entropy-weighted)
                suffix_starts = {}
                p_entropies = {}
                for i in active:
                    block_attempts[i] += 1
                    t = len(all_output_ids[i])
                    p_entropies[i] = (
                        np.diff(
                            all_raw_entropies[i],
                            prepend=0,
                        ).clip(min=0)
                        ** cut_power
                    )
                    p_entropy_sum = p_entropies[i].sum()
                    if p_entropy_sum > 0:
                        p_entropies[i] /= p_entropy_sum
                    else:
                        p_entropies[i] = np.ones(t) / t
                    suffix_starts[i] = np.random.choice(t, p=p_entropies[i])

                # Single batched proposal call with per-prompt sample_len
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
                    # Counted before the accept/reject decision below: a rejected
                    # proposal was still decoded and costs exactly as much as an
                    # accepted one.
                    all_gen_tokens[i] += prop_output.n_generated_tokens

                    ss = suffix_starts[i]
                    suffix_logprobs = all_logprobs[i][
                        ss : ss + len(prop_output.logprobs)
                    ]
                    suffix_raw_logprobs = all_raw_logprobs[i][
                        ss : ss + len(prop_output.raw_logprobs)
                    ]

                    # Reverse kernel entropy distribution over the full proposed chain
                    prop_entropies = (
                        all_raw_entropies[i][:ss] + prop_output.raw_entropies
                    )
                    prop_p_entropy = (
                        np.diff(
                            prop_entropies,
                            prepend=0,
                        ).clip(min=0)
                        ** cut_power
                    )
                    prop_p_entropy_sum = prop_p_entropy.sum()
                    if prop_p_entropy_sum > 0:
                        prop_p_entropy /= prop_p_entropy_sum
                    else:
                        prop_p_entropy = np.ones(len(prop_p_entropy)) / len(
                            prop_p_entropy
                        )

                    # power distribution ratio
                    power_log_ratio = (
                        np.sum(prop_output.raw_logprobs) - np.sum(suffix_raw_logprobs)
                    ) / temp
                    # proposal distribution ratio
                    proposal_log_ratio = np.sum(suffix_logprobs) - np.sum(
                        prop_output.logprobs
                    )
                    # cut-point correction ratio: based on the probability of choosing the cut-point
                    cut_log_ratio = np.log(prop_p_entropy[ss]) - np.log(
                        p_entropies[i][ss]
                    )
                    log_acceptance_ratio = (
                        power_log_ratio + proposal_log_ratio + cut_log_ratio
                    )
                    block_power[i].append(power_log_ratio)
                    block_proposal[i].append(proposal_log_ratio)
                    block_cut[i].append(cut_log_ratio)

                    acceptance_prob = min(1.0, float(np.exp(log_acceptance_ratio)))
                    accepted = np.random.rand() < acceptance_prob

                    if all_histories[i] is not None:
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
                            cut_probs=p_entropies[i],
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
                all_stats[i].avg_power_log_ratio.append(np.mean(block_power[i]))
                all_stats[i].avg_proposal_log_ratio.append(np.mean(block_proposal[i]))
                all_stats[i].avg_cut_log_ratio.append(np.mean(block_cut[i]))

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
                n_generated_tokens=all_gen_tokens[i],
            )
            for i in range(batch_size)
        ]
