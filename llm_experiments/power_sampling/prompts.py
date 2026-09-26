"""Prompt formatting for MATH500 and AIME26."""

from .constants import MATH_INSTRUCTION, MATH_REMINDER, MATH_SYSTEM_PROMPT

# The Qwen family are base models prompted with the raw string; the Phi family are
# instruction-tuned and get their own chat template.
QWEN_MODELS = ("qwen", "qwen_math", "qwen3_8b")
PHI_MODELS = ("phi35_instruct", "phi4_instruct")


def format_prompt(question, model, tokenizer):
    """Format a math question into a model-specific prompt string.

    Args:
        question: str — the math problem text.
        model: str — model identifier: "qwen", "qwen_math", "qwen3_8b" (raw prompt)
            or "phi35_instruct", "phi4_instruct" (chat template).
        tokenizer: HuggingFace tokenizer — used for apply_chat_template().

    Returns:
        format_str: str — the fully formatted prompt string ready for tokenization.
    """
    qwen_prompt = MATH_INSTRUCTION + "\n\n" + question + "\n\n" + MATH_REMINDER

    if model in QWEN_MODELS:
        return qwen_prompt

    if model in PHI_MODELS:
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": qwen_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    raise ValueError(f"Unknown model: {model}")
