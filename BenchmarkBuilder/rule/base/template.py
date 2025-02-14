PROMPT_NI_HINT_TEMPLATES = [
    'Kindly provide a number as the answer.',
    'Give a number as the answer.',
    'Give a numerical response.',
    'Offer a number as the answer.',
    'Provide a number as the answer.',
    'Please provide a numerical response.',
    'Please provide a number as the answer.',
    'Please reply with a number only.',
    'Reply with a number only.',
]

PROMPT_NI_CoT_HINT_TEMPLATE = """
Please solve the problem step by step. 
Show each intermediate thought process clearly and provide 
the final answer as a number only after completing the reasoning process.
"""

PROMPT_FV_HINT_TEMPLATES = [
    'Kindly provide a "yes" or "no" as the answer.',
    'Give a "yes" or "no" as the answer.',
    'Select "yes" or "no" as the answer.',
    'Offer a "yes" or "no" as the answer.',
    'Provide a "yes" or "no" as the answer.',
    'Please provide a "yes" or "no" as the answer.',
    'Please reply with a "yes" or "no" only.',
    'Reply with a "yes" or "no" only.',
]

PROMPT_FV_CoT_HINT_TEMPLATE = """
Please solve the problem step by step.
Show each intermediate thought process clearly and provide 
the final answer as "yes" or "no" only after completing the reasoning process.
"""

CAPTION_FV_AFFIRMATIVES = ['yes', 'true', 'correct', 'right', 'affirmative', 'positive']
CAPTION_FV_NEGATIVES = ['no', 'false', 'incorrect', 'wrong', 'negative']
