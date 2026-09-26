"""Power sampling with vLLM.

Modules are imported explicitly (e.g. ``from power_sampling.base import BaseSampler``).
This file deliberately imports nothing, so lightweight modules such as
``power_sampling.bon_utils`` and ``power_sampling.constants`` can be used without
torch or vLLM installed.

- base.py        BaseSampler: loads the model; batch_naive_temp (Standard / Low-Temperature)
- mcmc_cut.py    McmcCutSampler      (Entropy-Cut MH)
- mcmc_orig.py   McmcOrigSampler     (Uniform-Cut MH)
- smc.py         SmcSampler          (SMC)
- tmc.py         TmcSampler          (TMC)
- best_of_n.py   BestOfNSampler      (compute-matched Standard / Low-Temperature BoN)
- outputs.py     result/diagnostic containers (NaiveTempOutput, McmcOutput, McmcHistory, ...)
- tracker.py     LogitsTracker (raw log-probs / entropies from vLLM logits)
- utils.py       list helper
- prompts.py     format_prompt (MATH500 / AIME26)
- constants.py   prompt strings
- bon_utils.py   Best-of-N budget arithmetic (pure numpy)
"""
