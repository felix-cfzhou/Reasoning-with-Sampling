"""BaseSampler: owns the vLLM model and implements plain (temperature-scaled)
autoregressive sampling via `batch_naive_temp`.

The `naive_temp` arm calls `batch_naive_temp` at the low temperature `temp`
(= 1/alpha) and the `std` arm calls it at temperature 1.0. Every algorithm
sampler (MCMC, SMC, TMC, Best-of-N) takes a `BaseSampler` in its
constructor and generates through it, so one process loads the model once.

Requires the vLLM V0 engine (VLLM_USE_V1=0, set by the experiment drivers
before import): `LogitsTracker` relies on per-request `logits_processors`.
"""

import time

from tqdm import tqdm

from vllm import LLM, SamplingParams

from .outputs import NaiveTempOutput
from .tracker import LogitsTracker


class BaseSampler:
    """A wrapper around a VLLM interface to a HuggingFace causal LM for autoregressive sampling.

    Attributes:
        model_name: string name for a HuggingFace causal language model.
    """

    def __init__(self, model_name, max_model_len=8192, stop=None):

        hf_overrides = None
        disable_sliding_window = True
        assert max_model_len % 4096 == 0
        factor = max_model_len // 4096
        if model_name == "Qwen/Qwen2.5-Math-7B" and factor > 1:
            hf_overrides = {
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": 4096,
                    "type": "yarn",
                    "rope_type": "yarn",
                }
            }
            print(f"Overriding HF config with\n{hf_overrides}")
            disable_sliding_window = False

        self.model = LLM(
            model=model_name,
            gpu_memory_utilization=0.6,
            max_model_len=max_model_len,
            hf_overrides=hf_overrides,
            # important to maintain KV cache between iterations!
            enable_prefix_caching=True,
            disable_sliding_window=disable_sliding_window,
            trust_remote_code=True,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.stop = stop

    def batch_naive_temp(
        self,
        batch_context_ids,
        temp,
        sample_len,
        min_len=0,
        ignore_eos=False,
        seed=None,
        top_k=None,
        use_tqdm=True,
    ) -> list[NaiveTempOutput]:
        """Generate sequences for a batch of prompts in a single vLLM call.

        Args:
            batch_context_ids: List[List[int]] — list of token-id contexts to
                condition generation on.
            temp: Temperature parameter (= 1/alpha).
            sample_len: int or List[int] — number of new tokens to generate.
                If a single int, the same length is used for every prompt.
                If a list, must have one entry per prompt.
            min_len: Minimum number of tokens to generate per prompt (default 0).
            ignore_eos: If True, do not stop at the EOS token (default False).
            seed: Optional random seed forwarded to vLLM.

        Returns:
            results: List of NaiveTempOutput objects,
                one per prompt, in the same order as `batch_context_ids`.
        """
        _t0 = time.monotonic()

        # Normalise sample_len to a per-prompt list
        if isinstance(sample_len, int):
            sample_lens = [sample_len] * len(batch_context_ids)
        else:
            assert len(sample_len) == len(batch_context_ids), (
                f"batch_naive_temp: len(sample_len)={len(sample_len)} "
                f"!= len(batch_context_ids)={len(batch_context_ids)}"
            )
            sample_lens = sample_len

        # Each request needs its own LogitsTracker (called per-request by vLLM),
        # so we build per-request SamplingParams.
        trackers = []
        all_params = []

        for slen in sample_lens:
            tracker = LogitsTracker(top_k=top_k)
            trackers.append(tracker)
            all_params.append(
                SamplingParams(
                    n=1,
                    temperature=temp,
                    logprobs=0,
                    logits_processors=[tracker],
                    max_tokens=slen,
                    top_k=top_k if top_k is not None else -1,
                    min_tokens=min_len,
                    ignore_eos=ignore_eos,
                    detokenize=self.stop is not None,
                    seed=seed,
                    stop=self.stop,
                    include_stop_str_in_output=False,
                )
            )

        def custom_pbar(*args, **kwargs):
            return tqdm(*args, **kwargs, leave=False)

        # Single batched call to vLLM — all requests are scheduled together
        all_outputs = self.model.generate(
            prompt_token_ids=batch_context_ids,
            sampling_params=all_params,
            use_tqdm=custom_pbar if use_tqdm else False,
        )

        n_prompts = len(batch_context_ids)
        elapsed = (time.monotonic() - _t0) / n_prompts if n_prompts > 0 else 0.0

        results = []
        for output, tracker in zip(all_outputs, trackers):
            out = output.outputs[0]
            output_ids = list(out.token_ids)
            # Capture the true decode count BEFORE the phantom-EOS append below: when a
            # stop string fires we synthesise an EOS that the engine never generated.
            n_generated = len(output_ids)
            logprobs = [
                id_logprob_map[id].logprob
                for id_logprob_map, id in zip(out.logprobs, output_ids)
            ]
            tracker_output = tracker.get_output(output_ids)
            raw_logprobs = tracker_output.raw_logprobs
            raw_entropies = tracker_output.raw_entropies

            if (
                out.finish_reason == "stop"
                and output_ids[-1] != self.tokenizer.eos_token_id
            ):
                output_ids.append(self.tokenizer.eos_token_id)
                logprobs.append(0.0)
                raw_logprobs.append(0.0)
                raw_entropies.append(0.0)

            assert (
                len(output_ids)
                == len(logprobs)
                == len(raw_logprobs)
                == len(raw_entropies)
            )

            results.append(
                NaiveTempOutput(
                    output_ids,
                    logprobs,
                    raw_logprobs,
                    raw_entropies,
                    elapsed_time=elapsed,
                    n_generated_tokens=n_generated,
                )
            )

        return results
