"""Output and diagnostic containers returned by the samplers."""

from typing import Optional


class TrackerOutput:
    raw_logprobs: list[float]
    raw_entropies: list[float]

    def __init__(self, raw_logprobs, raw_entropies):
        self.raw_logprobs = raw_logprobs
        self.raw_entropies = raw_entropies


class McmcStats:
    accepts_per_block: list[int]
    attempts_per_block: list[int]
    avg_power_log_ratio: list[float]
    avg_proposal_log_ratio: list[float]
    avg_cut_log_ratio: list[float]
    temp: float
    top_k: int
    cut_power: float

    def __init__(self, temp=None, top_k=None, cut_power=None):
        self.accepts_per_block = []
        self.attempts_per_block = []
        self.avg_power_log_ratio = []
        self.avg_proposal_log_ratio = []
        self.avg_cut_log_ratio = []
        self.temp = temp
        self.top_k = top_k
        self.cut_power = cut_power

    def acceptance_rate(self):
        return sum(self.accepts_per_block) / sum(self.attempts_per_block)


class NaiveTempOutput:
    output_ids: list[int]
    logprobs: list[float]
    raw_logprobs: list[float]
    raw_entropies: list[float]
    elapsed_time: float
    n_generated_tokens: int

    def __init__(
        self,
        output_ids,
        logprobs,
        raw_logprobs,
        raw_entropies,
        elapsed_time,
        n_generated_tokens=None,
    ):
        self.output_ids = output_ids
        self.logprobs = logprobs
        self.raw_logprobs = raw_logprobs
        self.raw_entropies = raw_entropies
        self.elapsed_time = elapsed_time
        # Decode tokens the engine actually produced. Defaults to len(output_ids) for
        # the synthetic snapshots built by the MCMC history recorders; batch_naive_temp
        # passes the true count, which differs when a stop string appends a phantom EOS.
        self.n_generated_tokens = (
            len(output_ids) if n_generated_tokens is None else n_generated_tokens
        )


class McmcStepRecord:
    """A single MCMC step's diagnostic snapshot."""

    current: NaiveTempOutput
    cut_index: int
    cut_probs: list[float]
    proposal: NaiveTempOutput
    acceptance_prob: float
    accepted: bool
    output_tokens: Optional[list[str]]
    proposal_tokens: Optional[list[str]]

    def __init__(
        self,
        current,
        cut_index,
        cut_probs,
        proposal,
        acceptance_prob,
        accepted,
        output_tokens=None,
        proposal_tokens=None,
    ):
        self.current = current
        self.cut_index = cut_index
        self.cut_probs = cut_probs
        self.proposal = proposal
        self.acceptance_prob = acceptance_prob
        self.accepted = accepted
        self.output_tokens = output_tokens
        self.proposal_tokens = proposal_tokens

    def decode(self, tokenizer):
        """Populate output_tokens and proposal_tokens by decoding each token ID.

        Args:
            tokenizer: A HuggingFace-compatible tokenizer exposing
                ``batch_decode(List[int]) -> List[str]``.
        """
        self.output_tokens = tokenizer.batch_decode(self.current.output_ids)
        self.proposal_tokens = tokenizer.batch_decode(self.proposal.output_ids)


class McmcHistory:
    """Stores the full history of MCMC steps for diagnostic / analysis purposes.

    Steps are organised by block, mirroring the outer block / inner MCMC-step
    loop structure of the sampler.  Each entry records the state *before* the
    accept/reject decision, the proposal that was generated, the probability
    used to accept it, and whether it was actually accepted.

    Attributes:
        blocks: List[List[McmcStepRecord]] — one inner list per block, each
            containing one record per MCMC step within that block.
    """

    blocks: list[list[McmcStepRecord]]

    def __init__(self):
        self.blocks = []

    def new_block(self):
        """Start a new block. Subsequent ``record_step`` calls will append to
        this block until ``new_block`` is called again."""
        self.blocks.append([])

    def record_step(
        self,
        current,
        cut_index,
        cut_probs,
        proposal,
        acceptance_prob,
        accepted,
    ):
        """Append a new step record to the current block.

        Args:
            current: NaiveTempOutput — the current chain state as a
                NaiveTempOutput (output_ids, logprobs, raw_logprobs, raw_entropies).
            cut_index: int — the sampled cut-point index.
            cut_probs: array-like of float — the probability distribution over
                positions used to sample the cut-point.
            proposal: NaiveTempOutput — the generated proposal.
            acceptance_prob: float — min(1, exp(log_acceptance_ratio)).
            accepted: bool — whether the proposal was accepted.
        """
        if not self.blocks:
            self.blocks.append([])
        self.blocks[-1].append(
            McmcStepRecord(
                current=current,
                cut_index=cut_index,
                cut_probs=list(cut_probs),
                proposal=proposal,
                acceptance_prob=min(1.0, max(0.0, acceptance_prob)),
                accepted=accepted,
            )
        )

    @property
    def num_blocks(self):
        """Number of blocks recorded so far."""
        return len(self.blocks)

    @property
    def current_block(self):
        """The list of step records for the most recent block."""
        return self.blocks[-1] if self.blocks else []

    @property
    def all_steps(self):
        """Flat iterator over every step record across all blocks."""
        return [step for block in self.blocks for step in block]

    def __len__(self):
        """Total number of step records across all blocks."""
        return sum(len(b) for b in self.blocks)

    def __getitem__(self, block_idx):
        """Index by block: ``history[i]`` returns the list of steps for block *i*."""
        return self.blocks[block_idx]

    @staticmethod
    def _format_text_block(label: str, text: str, width: int = 72) -> list[str]:
        """Render *text* inside a labelled Unicode box.

        The top border is:  ┌─ <label> ─────┐
        Each content line:  │ <line>
        The bottom border:  └───────────────┘

        Actual newlines in *text* are preserved so the output is human-readable
        rather than showing escaped ``\\n`` sequences.  Long lines are left
        as-is (they will simply overflow the box visually).
        """
        inner = width - 2  # width of the space between the two border chars
        label_str = f"─ {label} "
        top = "┌" + label_str + "─" * max(0, inner - len(label_str)) + "┐"
        bottom = "└" + "─" * inner + "┘"
        result = [top]
        for line in text.split("\n"):
            result.append("│ " + line)
        result.append(bottom)
        return result

    def __str__(self):
        lines = []
        lines.append(f"McmcHistory: {self.num_blocks} blocks, {len(self)} total steps")
        lines.append("=" * 60)
        for b_idx, block in enumerate(self.blocks):
            lines.append(f"\n--- Block {b_idx} ({len(block)} steps) ---")
            for s_idx, step in enumerate(block):
                assert step.output_tokens is not None, (
                    f"output_tokens is None for block {b_idx}, step {s_idx}. "
                    "Call McmcHistory.batch_decode() first."
                )
                assert step.proposal_tokens is not None, (
                    f"proposal_tokens is None for block {b_idx}, step {s_idx}. "
                    "Call McmcHistory.batch_decode() first."
                )
                status = "✓ ACCEPTED" if step.accepted else "✗ rejected"
                lines.append(
                    f"  Step {s_idx}  [{status}]  (accept prob: {step.acceptance_prob:.4f})"
                )
                lines.append(f"    Cut index : {step.cut_index}")

                # Insert CUT_MARKER into output tokens before joining
                output_tokens_with_cut = list(step.output_tokens)
                output_tokens_with_cut.insert(step.cut_index, McmcHistory.CUT_MARKER)
                output_str = "".join(output_tokens_with_cut)
                proposal_str = "".join(step.proposal_tokens)

                lines.extend(
                    "  " + l
                    for l in McmcHistory._format_text_block("Current", output_str)
                )
                lines.extend(
                    "  " + l
                    for l in McmcHistory._format_text_block("Proposal", proposal_str)
                )
        return "\n".join(lines)

    def get_latex_str(self, mode: str = "entropy", max_display_tokens: int = 144):
        """Render the MCMC history as a LaTeX string.

        Args:
            mode: Colormap mode for token heat values.
                ``"entropy"``        — heat = raw Shannon entropy of the token's
                                       conditional distribution (default behaviour).
                ``"entropy_change"`` — heat = max(0, entropy_t − entropy_{t-1}),
                                       i.e. the *increase* in entropy from the
                                       previous proposal token to the current one.
                                       The first token always falls back to its
                                       raw entropy (no predecessor to compare).
            max_display_tokens: Maximum number of tokens to display before/after
                the cut index in the current output, and at the start of the
                proposal. Omitted tokens are replaced with ``...``.
        """

        def format_latex_token(tok, heat_value):
            num_newlines = tok.count("\n")
            tok = tok.replace("\n", "↩")
            # Find a character that is NOT inside the token to use as the boundary
            if "|" not in tok:
                delimiter = "|"
            elif "!" not in tok:
                delimiter = "!"
            elif "+" not in tok:
                delimiter = "+"
            else:
                delimiter = "*"  # Fallback

            # Format with the chosen delimiter instead of {}
            formatted = f"\\token[{heat_value:.4f}]{delimiter}{tok}{delimiter}"
            formatted += "\\newline" * num_newlines
            return formatted

        lines = []
        lines.append(
            f"\\section*{{McmcHistory: {self.num_blocks} block(s), {len(self)} total steps}}"
        )
        for b_idx, block in enumerate(self.blocks):
            for s_idx, step in enumerate(block):
                assert step.output_tokens is not None, (
                    f"output_tokens is None for block {b_idx}, step {s_idx}. "
                    "Call McmcHistory.batch_decode() first."
                )
                assert step.proposal_tokens is not None, (
                    f"proposal_tokens is None for block {b_idx}, step {s_idx}. "
                    "Call McmcHistory.batch_decode() first."
                )
                status = "Accepted" if step.accepted else "Rejected"
                lines.append(
                        f"\\begin{{twopartbox}}{{Cut Index {step.cut_index} | Acceptance Probability {step.acceptance_prob:.4f}}}{{Block {b_idx+1} Step {s_idx+1}}}{{Proposal ({status})}}"
                )

                # Format current output tokens with latex
                output_latex_tokens = []
                for tok, entropy in zip(step.output_tokens, step.current.raw_entropies):
                    output_latex_tokens.append(format_latex_token(tok, entropy))

                cut_index = step.cut_index
                before = output_latex_tokens[:cut_index]
                after = output_latex_tokens[cut_index:]
                before_omitted = len(before) > max_display_tokens
                after_omitted = len(after) > max_display_tokens
                before = before[-max_display_tokens:]
                after = after[:max_display_tokens]
                output_pieces = []
                if before_omitted:
                    output_pieces.append(McmcHistory.TRUNC_MARKER)
                output_pieces.extend(before)
                output_pieces.append(McmcHistory.CUT_MARKER)
                output_pieces.extend(after)
                if after_omitted:
                    output_pieces.append(McmcHistory.TRUNC_MARKER)
                output_str = "\\allowbreak".join(output_pieces)

                # Format proposal tokens with latex.
                # In entropy_change mode the heat value is max(0, H_t - H_{t-1}),
                # i.e. the increase in entropy from the previous proposal token.
                # The first token (pos==0) has no predecessor and falls back to
                # raw entropy.
                proposal_latex_tokens = []
                for pos, (tok, prop_entropy) in enumerate(
                    zip(step.proposal_tokens, step.proposal.raw_entropies)
                ):
                    if mode == "entropy_change" and pos > 0:
                        prev_entropy = step.proposal.raw_entropies[pos - 1]
                        heat = max(0.0, prop_entropy - prev_entropy)
                    else:
                        heat = prop_entropy
                    proposal_latex_tokens.append(format_latex_token(tok, heat))
                proposal_omitted = len(proposal_latex_tokens) > max_display_tokens
                proposal_latex_tokens = proposal_latex_tokens[:max_display_tokens]
                if proposal_omitted:
                    proposal_latex_tokens.append(McmcHistory.TRUNC_MARKER)
                proposal_str = "\\allowbreak".join(proposal_latex_tokens)

                lines.append(output_str)
                lines.append(r"\tcblower")
                lines.append(proposal_str)
                lines.append(r"\end{twopartbox}")
        return "\n".join(lines)

    CUT_MARKER = r"\cut{[CUT]}"
    TRUNC_MARKER = r"[...]"

    def batch_decode(self, tokenizer):
        """Decode all stored token sequences in this history.

        For every :class:`McmcStepRecord`, this calls ``step.decode(tokenizer)``
        to populate ``output_tokens`` and ``proposal_tokens``.

        Args:
            tokenizer: A HuggingFace-compatible tokenizer exposing
                ``batch_decode(List[int]) -> List[str]``.
        """
        for step in self.all_steps:
            step.decode(tokenizer)


class McmcOutput:
    output_ids: list[int]
    logprobs: list[float]
    raw_logprobs: list[float]
    raw_entropies: list[float]
    stats: McmcStats
    history: Optional[McmcHistory]
    elapsed_time: float
    n_generated_tokens: int

    def __init__(
        self,
        output_ids,
        logprobs,
        raw_logprobs,
        raw_entropies,
        stats,
        elapsed_time,
        history=None,
        n_generated_tokens=0,
    ):
        self.output_ids = output_ids
        self.logprobs = logprobs
        self.raw_logprobs = raw_logprobs
        self.raw_entropies = raw_entropies
        self.stats = stats
        self.elapsed_time = elapsed_time
        self.history = history
        # Total decode tokens across the whole chain, including tokens generated for
        # rejected proposals and tokens discarded after an EOS truncation.
        self.n_generated_tokens = n_generated_tokens


class BestOfNOutput:
    """The single candidate selected by Best-of-N, plus its budget accounting."""

    output_ids: list[int]
    logprobs: list[float]
    raw_logprobs: list[float]
    raw_entropies: list[float]
    avg_logprob: float  # selection statistic of the winner
    n_candidates: int  # candidates drawn and eligible for selection
    n_generated_tokens: int  # decode tokens of those candidates; >= token_budget
    token_budget: int
    stop_reason: str  # "budget" | "max_n"
    elapsed_time: float

    def __init__(
        self,
        output_ids,
        logprobs,
        raw_logprobs,
        raw_entropies,
        avg_logprob,
        n_candidates,
        n_generated_tokens,
        token_budget,
        stop_reason,
        elapsed_time,
    ):
        self.output_ids = output_ids
        self.logprobs = logprobs
        self.raw_logprobs = raw_logprobs
        self.raw_entropies = raw_entropies
        self.avg_logprob = avg_logprob
        self.n_candidates = n_candidates
        self.n_generated_tokens = n_generated_tokens
        self.token_budget = token_budget
        self.stop_reason = stop_reason
        self.elapsed_time = elapsed_time


class SmcParticle:
    output_ids: list[int]
    logprobs: list[float]
    raw_logprobs: list[float]
    raw_entropies: list[float]

    def __init__(self, output_ids, logprobs, raw_logprobs, raw_entropies):
        self.output_ids = output_ids
        self.logprobs = logprobs
        self.raw_logprobs = raw_logprobs
        self.raw_entropies = raw_entropies


class SmcOutput:
    sampled_particle: SmcParticle
    log_weight: float
    all_particles: list[SmcParticle]
    log_weights: list[float]
    elapsed_time: float

    def __init__(
        self,
        sampled_particle,
        log_weight,
        all_particles,
        log_weights,
        elapsed_time,
    ):
        self.sampled_particle = sampled_particle
        self.log_weight = log_weight
        self.all_particles = all_particles
        self.log_weights = log_weights
        self.elapsed_time = elapsed_time
