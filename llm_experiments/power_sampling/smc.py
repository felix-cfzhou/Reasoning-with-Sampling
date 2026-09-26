"""Sequential Monte Carlo power sampling (the `smc` arm)."""

import gc
import time

import numpy as np
import scipy.special
import torch
from tqdm import tqdm

from .outputs import SmcOutput, SmcParticle
from .utils import unflatten_list


class SmcSampler:
    """Blockwise SMC with ESS-triggered systematic resampling, targeting p^(1/temp)."""

    def __init__(self, base):
        self.base = base

    def batch_smc_power_sample(
        self,
        batch_prompt_ids,
        num_particles,
        max_new_tokens,
        temp=1.0,
        top_k=None,
        batch_size=4,
        block_size=32,
    ) -> list[SmcOutput]:
        """
        Run SMC Power Sampling for a batch of prompts, processing them in mini-batches.

        Args:
            batch_prompt_ids: List of List of tokenized prompt IDs.
            num_particles: Number of particles to use per prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temp: Temperature for sampling.
            top_k: Top-k for sampling.
            batch_size: Mini-batch size for processing prompts.
            block_size: Target size for jump_size.

        Returns:
            List[SmcOutput] object, one per prompt in `batch_prompt_ids`.
        """
        all_outputs = []
        for i in tqdm(
            range(0, len(batch_prompt_ids), batch_size),
            desc="SMC mini-batches",
            leave=False,
        ):
            chunk_prompt_ids = batch_prompt_ids[i : i + batch_size]
            chunk_outputs = self._batch_smc_power_sample(
                chunk_prompt_ids,
                num_particles,
                max_new_tokens,
                temp,
                top_k,
                block_size,
            )
            all_outputs.extend(chunk_outputs)

            gc.collect()  # python garbage collection
            torch.cuda.empty_cache()  # torch garbage collection

        return all_outputs

    def _batch_smc_power_sample(
        self,
        batch_prompt_ids,
        num_particles,
        max_new_tokens,
        temp=1.0,
        top_k=None,
        block_size=16,
    ):
        """
        Run SMC Power Sampling for a batch of prompts.
        Uses systematic resampling
        (stratified with a single u0 offset) triggered by ESS < N/2,
        and un-normalized log weights for final particle selection.

        Args:
            batch_prompt_ids: List of List of tokenized prompt IDs.
            num_particles: Number of particles to use per prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temp: Temperature for sampling.
            top_k: Top-k for sampling.
            block_size: Target size for jump_size.

        Returns:
            List[SmcOutput] object, one per prompt in `batch_prompt_ids`.
        """
        assert (
            max_new_tokens % block_size == 0
        ), f"_batch_smc_power_sample: max_new_tokens {max_new_tokens} must be divisible by block_size {block_size}"
        _t0 = time.monotonic()
        jump_size = block_size
        num_blocks = max_new_tokens // jump_size
        batch_size = len(batch_prompt_ids)

        # Flattened list of prompt_ids for initial generation
        flattened_prompts = [
            prompt for prompt in batch_prompt_ids for _ in range(num_particles)
        ]

        # --- Step 1: Extend the sequence by jump_size tokens ---
        init_outputs = self.base.batch_naive_temp(
            flattened_prompts,
            temp=temp,
            sample_len=jump_size,
            ignore_eos=False,
            top_k=top_k,
            use_tqdm=False,
        )

        # Reshape the 1D list of outputs into a 2D list of shape (batch_size, num_particles)
        initial_lens = [num_particles] * batch_size
        all_output_ids = unflatten_list(
            [out.output_ids for out in init_outputs], initial_lens
        )
        all_logprobs = unflatten_list(
            [out.logprobs for out in init_outputs], initial_lens
        )
        all_raw_logprobs = unflatten_list(
            [out.raw_logprobs for out in init_outputs], initial_lens
        )
        all_raw_entropies = unflatten_list(
            [out.raw_entropies for out in init_outputs], initial_lens
        )

        # --- Step 2: Run SMC for each block ---
        for _block in tqdm(range(1, num_blocks), desc="SMC blocks", leave=False):

            # For each prompt: compute cumulative log weights, resample if ESS < N/2,
            # then filter active particles and extend.
            for i in range(batch_size):
                # Compute cumulative log weights for all particles
                all_log_weights_i = [
                    np.sum(all_raw_logprobs[i][j]) / temp - np.sum(all_logprobs[i][j])
                    for j in range(num_particles)
                ]

                # Systematic resampling if ESS < N/2
                all_probs_i = scipy.special.softmax(all_log_weights_i)
                eff_sample_size = 1.0 / np.sum(all_probs_i**2)
                if eff_sample_size < num_particles / 2:
                    u0 = np.random.uniform(0, 1)
                    positions = (u0 + np.arange(num_particles)) / num_particles
                    cumulative_weights = np.cumsum(all_probs_i)
                    ancestors = np.searchsorted(cumulative_weights, positions).clip(
                        max=num_particles - 1
                    )

                    # Copy the states of the ancestors
                    all_output_ids[i] = [list(all_output_ids[i][a]) for a in ancestors]
                    all_logprobs[i] = [list(all_logprobs[i][a]) for a in ancestors]
                    all_raw_logprobs[i] = [
                        list(all_raw_logprobs[i][a]) for a in ancestors
                    ]
                    all_raw_entropies[i] = [
                        list(all_raw_entropies[i][a]) for a in ancestors
                    ]

            # Determine active particles across all prompts
            all_context_ids = []
            active_map = []  # list of (prompt_idx, particle_idx) for active particles
            for i in range(batch_size):
                for j in range(num_particles):
                    if self.base.tokenizer.eos_token_id not in all_output_ids[i][j]:
                        all_context_ids.append(
                            batch_prompt_ids[i] + all_output_ids[i][j]
                        )
                        active_map.append((i, j))

            if not all_context_ids:
                break

            naive_temp_outputs = self.base.batch_naive_temp(
                all_context_ids,
                temp=temp,
                sample_len=jump_size,
                ignore_eos=False,
                top_k=top_k,
                use_tqdm=False,
            )

            assert len(active_map) == len(naive_temp_outputs), (
                f"batch_smc_power_sample: len(active_map)={len(active_map)} "
                f"!= len(naive_temp_outputs)={len(naive_temp_outputs)}"
            )

            # Extend active particles with new tokens
            for (i, j), naive_output in zip(active_map, naive_temp_outputs):
                all_output_ids[i][j].extend(naive_output.output_ids)
                all_logprobs[i][j].extend(naive_output.logprobs)
                all_raw_logprobs[i][j].extend(naive_output.raw_logprobs)
                all_raw_entropies[i][j].extend(naive_output.raw_entropies)

        # Final selection over ALL particles using cumulative log weights
        elapsed = (time.monotonic() - _t0) / batch_size if batch_size > 0 else 0.0
        all_smc_outputs = []
        for i in range(batch_size):
            output_particles = [
                SmcParticle(
                    all_output_ids[i][j],
                    all_logprobs[i][j],
                    all_raw_logprobs[i][j],
                    all_raw_entropies[i][j],
                )
                for j in range(num_particles)
            ]
            all_log_weights_i = [
                np.sum(all_raw_logprobs[i][j]) / temp - np.sum(all_logprobs[i][j])
                for j in range(num_particles)
            ]
            all_probs_i = scipy.special.softmax(all_log_weights_i)
            sampled_particle_j = np.random.choice(
                num_particles, size=1, replace=False, p=all_probs_i
            ).item()

            smc_output = SmcOutput(
                output_particles[sampled_particle_j],
                all_log_weights_i[sampled_particle_j],
                output_particles,
                all_log_weights_i,
                elapsed_time=elapsed,
            )
            all_smc_outputs.append(smc_output)

        return all_smc_outputs
