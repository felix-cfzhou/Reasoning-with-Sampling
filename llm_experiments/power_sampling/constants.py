"""Prompt strings used in the paper.

The MATH strings are concatenated, never passed through ``str.format``, so their
``{{}}`` reaches the model verbatim. The GPQA template is filled with ``format``.
"""

# MATH500 and AIME26
MATH_INSTRUCTION = "Can you solve the following math problem? Please reason step by step, and put your final answer within \\boxed{{}}."
MATH_REMINDER = "Remember to present your final answer within \\boxed{{}}!"
MATH_SYSTEM_PROMPT = "You are an AI math expert."

# GPQA Diamond
GPQA_QUERY_TEMPLATE = "Answer the following multiple choice question. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
