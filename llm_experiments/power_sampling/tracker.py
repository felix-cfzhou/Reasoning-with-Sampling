"""LogitsTracker: a vLLM (V0) logits processor that records the base model's raw
log-probabilities and entropies before temperature is applied.
"""

import torch

from .outputs import TrackerOutput


class LogitsTracker:
    def __init__(self, top_k=None):
        self.prev_logits = None  # previous step's logits tensor
        self.raw_logprobs = []
        self.raw_entropies = []
        self.past_token_ids = None
        self.top_k = top_k

    def _logprob_and_entropy(self, logits_float, chosen_id):
        """Compute the log-prob of `chosen_id` and the conditional Shannon entropy
        H(X_t | X_{<t}) = -sum_x p(x) log p(x) from `logits_float` (dense).

        When self.top_k is set, the normalizing constant (logsumexp), logprob,
        and entropy are all computed over the top-k logits only.

        Returns:
            (logprob, entropy)
        """
        if self.top_k is not None:
            topk_vals, _ = torch.topk(logits_float, self.top_k)
            logsumexp = topk_vals.logsumexp(dim=0)
            logprob = (logits_float[chosen_id] - logsumexp).item()
            token_logprobs = topk_vals - logsumexp
        else:
            logsumexp = logits_float.logsumexp(dim=0)
            logprob = (logits_float[chosen_id] - logsumexp).item()
            token_logprobs = logits_float - logsumexp
        token_probs = token_logprobs.exp()
        entropy = torch.special.entr(token_probs).sum().item()

        return logprob, entropy

    def __call__(self, past_token_ids, logits):
        # 1. If we have logits from a previous step, the last ID in 'token_ids'
        # is the token that was just chosen using those logits.
        if self.prev_logits is not None and len(past_token_ids) > 0:
            last_chosen_id = past_token_ids[-1]
            prev_float = self.prev_logits.float()
            logprob, entropy = self._logprob_and_entropy(prev_float, last_chosen_id)
            self.raw_logprobs.append(logprob)
            self.raw_entropies.append(entropy)

        # 2. Store current logits (dense) for the next token's turn.
        self.prev_logits = logits.clone()
        self.past_token_ids = list(past_token_ids)

        return logits

    def get_output(self, token_ids):
        """Finalise tracking and return a TrackerOutput with logprobs and entropies.

        When generation stops (e.g. EOS) the logits processor is not called
        again, so the logprob/entropy for the very last token is never recorded.
        Compute it here from the stored prev_logits.
        """
        if len(self.raw_logprobs) == len(token_ids) - 1:
            last_chosen_id = token_ids[-1]
            prev_float = self.prev_logits.float()
            logprob, entropy = self._logprob_and_entropy(prev_float, last_chosen_id)
            self.raw_logprobs.append(logprob)
            self.raw_entropies.append(entropy)
            self.past_token_ids.append(last_chosen_id)

        assert len(self.raw_logprobs) == len(
            token_ids
        ), "LogitsTracker: number of stored logprobs does not match number of generated token ids."
        assert (
            self.past_token_ids == token_ids
        ), f"get_output: {self.past_token_ids} and {token_ids} do not match!"

        return TrackerOutput(self.raw_logprobs, self.raw_entropies)
