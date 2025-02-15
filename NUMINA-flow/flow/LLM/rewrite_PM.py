import random
import re
import string
from typing import Union, Dict

from .base import LLMBasedQRewriter
from ..utils.io import parse_json_text


class PromptMatchingRewriter(LLMBasedQRewriter):
    """Prompt matching (multiple-choice question) generator implementation"""

    def __init__(
            self,
            n_options: int = 3,
            enforce_balanced_options: bool = False,
            **kwargs
    ):
        # extract MCQ-specific parameters before calling parent init
        self.n_options = n_options
        if self.n_options < 2:
            raise ValueError('Number of options must be at least 2')
        self.is_evenly_shuffled_options = enforce_balanced_options

        # Call parent init with remaining kwargs
        super().__init__(
            question_json_file=kwargs['question_json_file'],
            rewrite_question_type='LLM_rewrite-PM',
            llm_model=kwargs.get('llm_model', 'qwen2.5:72b'),
            llm_backend=kwargs.get('llm_backend', 'ollama'),
            output_path=kwargs.get('output_path', './output/')
        )

    def _rewrite_question(
            self,
            src_question: str,
            src_answer: str,
            question_set_idx: int,
            **kwargs
    ) -> dict[str, Union[str, int, float]]:
        """Rewrite the question as a multiple-choice question"""
        # cleanup answer text
        src_answer_cleanup = src_answer.rstrip('.').strip()
        # generate rewrite caption
        if self.is_evenly_shuffled_options:
            preset_rewritten_option = string.ascii_uppercase[question_set_idx % self.n_options]
        else:
            preset_rewritten_option = random.choice(string.ascii_uppercase[:self.n_options])

        # generate the LLM prompt
        llm_prompt = str(
            f'<Introduction>\n'
            f'Please rewrite a Short Answer Question (SAQ) into a multiple-choice question (MCQ) format '
            f'with {self.n_options} options\n\n'
            f'<Input>\n'
            f' - Original SAQ: "{src_question}"\n'
            f' - Original Answer: "{src_answer_cleanup}"\n'
            f' - Expected Correct Option: {preset_rewritten_option}\n'
            f' - Number of Options: {self.n_options}\n\n'
            f'<Task Description>\n'
            f'1. Convert the SAQ into a clear and concise MCQ format.\n'
            f'2. Generate {self.n_options - 1} incorrect options as distractors:\n'
            f'   - Keep them the same type as the correct answer '
            f'     (e.g., if the answer is a noun, distractors should also be nouns).\n'
            f'   - Ensure all options are plausible but incorrect.\n'
            f'   - Avoid synonyms, too obvious wrong choices, or options that give away the answer.\n'
            f'3. Ensure that the correct answer is placed exactly as provided '
            f'   (including spelling mistakes) at the correct option label.\n'
            f'4. Format the options as follows:\n'
            f'   - "A) Option_A  B) Option_B  C) Option_C ..."\n'
            f'   - Ensure the correct answer is placed at the provided correct option label.\n'
            f'5. Insert a answering hint, suggesting to answer with the correct option letter.\n\n'
            f'<Output>\n'
            f'Return only a JSON object with exactly these keys:\n'
            f'  - \"prompt\": the rewritten MCQ\n'
            f'  - \"caption\": the correct option label (e.g., "A")\n'
            f'Do not include any additional text or explanation outside this JSON object.'
            f'<Example>\n'
            f'Input (Note: The original answer misspells "Paris" as "Parris"):\n'
            f'- SAQ: "What is the capital of France?"\n'
            f'- Answer: "Parris"\n'
            f'- Expected Correct Option: "B"\n'
            f'- Number of Options: 4\n\n'
            f'A valid output would be:\n'
            f'{{\n'
            f'  "prompt": "What is the capital of France? Answer the question with the correct option letter.'
            f' A) Berlin B) Parris C) London D) Rome",\n'
            f'  "caption": "B"\n'
            f'}}'
        )
        # get the rewritten question from the LLM model
        rewritten_question_dict = parse_json_text(
            self._chat_with_llm(llm_prompt), ['prompt', 'caption'])

        return {
            'meta': {
                'src_prompt': src_question,
                'src_caption': src_answer,
                'llm': {
                    'model': self.llm_model,
                    'backend': self.llm_backend
                }
            },
            'prompt': rewritten_question_dict['prompt'],
            'caption': preset_rewritten_option,
            'CoT_caption': f'<{preset_rewritten_option}>',
            'ref_captions': [preset_rewritten_option],
        }

    def _validate_rewritten_question(
            self,
            rewrite_output_dict: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]]
    ) -> bool:
        """Validate the rewritten multiple-choice question"""

        meta = rewrite_output_dict['meta']
        # extract source SAQ and correct answer
        src_caption = meta['src_caption'].rstrip('.').strip()
        preset_option_label = rewrite_output_dict['caption']

        # extract MCQ question and options
        options_pattern = re.compile(r'([A-Z])\)\s*(.*?)\s*(?=[A-Z]\)|$)')
        mcq_option_dict = {
            option: value.strip() for option, value in options_pattern.findall(rewrite_output_dict['prompt'])
        }
        mcq_main_body = options_pattern.sub('', rewrite_output_dict['prompt']).strip()

        # check if there are exactly n_options
        if len(mcq_option_dict) != self.n_options:
            raise ValueError(
                f'Possibly Invalid reply: '
                f'Expected {self.n_options} options, but found {len(mcq_option_dict)}: {mcq_option_dict}'
            )

        # check if the correct answer is placed at the preset correct option label
        if mcq_option_dict.get(preset_option_label, None) != src_caption:
            raise ValueError(
                f'Possibly Invalid reply: '
                f'Expected "{src_caption}" at option "{preset_option_label}", '
                f'but found "{mcq_option_dict.get(preset_option_label, None)}"'
            )

        # check if the correct answer is not present in the main body of the question
        if src_caption.lower() in mcq_main_body.lower():
            raise ValueError(
                f'Possibly Invalid reply: '
                f'Expected the correct answer "{src_caption}" to NOT appear in the main body of the question'
            )

        distractors = [value for label, value in mcq_option_dict.items() if label != preset_option_label]
        # check if distractors are distinct
        if len(set(distractors)) != len(distractors):
            raise ValueError(
                f'Possibly Invalid reply:'
                f'Expected {self.n_options - 1} distinct distractors, but found: {distractors}'
            )

        # check if the correct answer is not too similar to any distractor
        for distractor in distractors:
            if distractor.lower().startswith(src_caption.lower()) or distractor.lower().endswith(src_caption.lower()):
                raise ValueError(
                    f'Possibly Invalid reply: '
                    f'Expected distractor {distractor} to not start or end with the correct answer "{src_caption}"'
                )

        return True
