import random
from typing import Union, Dict

from .base import LLMBasedQRewriter
from ..utils.io import parse_json_text


class FactValidationRewriter(LLMBasedQRewriter):
    """Fact validation (True/False question) generator implementation"""

    def __init__(
            self,
            enforce_balanced_boolean: bool = False,
            **kwargs
    ):
        # extract TFQ-specific parameters before calling parent init
        self.enforce_balanced_boolean = enforce_balanced_boolean

        # Call parent init with remaining kwargs
        super().__init__(
            question_json_file=kwargs['question_json_file'],
            rewrite_question_type='LLM_rewrite-FV',
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
        """Rewrite the question as a true/false question"""
        # cleanup answer text
        src_answer_cleanup = src_answer.rstrip('.').strip()
        # generate rewrite caption
        if self.enforce_balanced_boolean:
            preset_rewritten_boolean = True if question_set_idx % 2 == 0 else False
        else:
            preset_rewritten_boolean = random.choice([True, False])

        affirmative_words = ['yes', 'true', 'correct', 'agree']
        negative_words = ['no', 'false', 'incorrect', 'disagree']
        # choose from pairs of words
        affirmative_word, negative_word = random.choice([*zip(affirmative_words, negative_words)])

        # generate the LLM prompt
        llm_prompt = str(
            f'<Introduction>\n'
            f'Please rewrite a Short Answer Question (SAQ) into a True/False Question (TFQ) format, '
            f'and also produce its contrapositive version.\n\n'
            f'<Input>\n'
            f' - Original SAQ: "{src_question}"\n'
            f' - Original Answer: "{src_answer_cleanup}"\n'
            f' - Boolean Indicator: {preset_rewritten_boolean}\n'
            f' - Answer Options: "{affirmative_word}" (for True) / "{negative_word}" (for False)\n\n'
            f'<Task Description>\n'
            f'1. Convert the SAQ into a clear, factual statement incorporating the given answer.\n'
            f'2. Rewrite the statement into a TFQ based on the Boolean Indicator:\n'
            f'   - If True → Keep the statement affirmative (e.g., "Bob sits next to Alice.")\n'
            f'   - If False → Negate the statement (e.g., "Bob does not sit next to Alice.")\n'
            f'3. Append the answer options at the end as: "Is this correct? Answer with {affirmative_word} or {negative_word}."\n'
            f'4. Generate the contrapositive TFQ by logically inverting the original statement while keeping the same answer options.\n'
            f'5. Ensure that both TFQs are logically and grammatically correct.\n\n'
            f'<Output>\n'
            f'Return only a JSON object with exactly these keys:\n'
            f'  - \"prompt\": the rewritten TFQ\n'
            f'  - \"caption\": the answer option corresponding to the preset boolean indicator\n'
            f'  - \"cp_prompt\": the contrapositive version of the rewritten TFQ\n'
            f'  - \"cp_caption\": the answer option corresponding to the negation of the preset boolean indicator\n'
            f'Do not include any additional text or explanation outside this JSON object.'
            f'<Example>\n'
            f'Input:\n'
            f'- SAQ: "Who sits next to Alice?"\n'
            f'- Answer: "Bob"\n'
            f'- Boolean Indicator: False\n'
            f'- Answer Options: "{affirmative_word}" (True) / "{negative_word}" (False)\n\n'
            f'A valid output would be:\n'
            f'{{\n'
            f'  "prompt": "Bob does not sit next to Alice. Is this correct? Answer with {affirmative_word} or {negative_word}.",\n'
            f'  "caption": "{negative_word}",\n'
            f'  "cp_prompt": "Bob sits next to Alice. Is this correct? Answer with {affirmative_word} or {negative_word}.",\n'
            f'  "cp_caption": "{affirmative_word}"\n'
            f'}}'
        )

        # get the rewritten question from the LLM model
        rewritten_question_dict = parse_json_text(
            self._chat_with_llm(llm_prompt), ['prompt', 'caption', 'cp_prompt', 'cp_caption'])

        return {
            'meta': {
                'src_prompt': src_question,
                'src_caption': src_answer,
                'llm': {
                    'model': self.llm_model,
                    'backend': self.llm_backend
                },
                'preset_boolean': preset_rewritten_boolean,
                'preset_affirmative_word': affirmative_word,
                'preset_negative_word': negative_word,
            },
            # original proposition
            'prompt': rewritten_question_dict['prompt'],
            'caption': affirmative_word if preset_rewritten_boolean else negative_word,
            'CoT_caption': f'<{affirmative_word if preset_rewritten_boolean else negative_word}>',
            'ref_captions': affirmative_words if preset_rewritten_boolean else negative_words,
            # contrapositive proposition
            'cp_prompt': rewritten_question_dict['cp_prompt'],
            'cp_caption': affirmative_word if not preset_rewritten_boolean else negative_word,
            'CoT_cp_caption': f'<{affirmative_word if not preset_rewritten_boolean else negative_word}>',
            'ref_cp_captions': affirmative_words if not preset_rewritten_boolean else negative_words,
        }

    def _validate_rewritten_question(
            self,
            rewrite_output_dict: Dict[str, Union[str, int, float, Dict[str, Union[str, int, float]]]]
    ) -> bool:
        """Validate the rewritten true/false question"""
        meta = rewrite_output_dict['meta']
        # extract boolean & answer option settings
        preset_boolean = meta['preset_boolean']
        affirmative_word = meta['preset_affirmative_word']
        negative_word = meta['preset_negative_word']

        # extract proposition & contrapositive proposition
        prompt = rewrite_output_dict['prompt']
        caption = rewrite_output_dict['caption']
        cp_prompt = rewrite_output_dict['cp_prompt']
        cp_caption = rewrite_output_dict['cp_caption']

        # check if the correct boolean word pair is used somewhere in both TFQs
        for prompt_text in [prompt, cp_prompt]:
            if not any(word in prompt_text for word in [affirmative_word, negative_word]):
                raise ValueError(
                    f'Possibly Invalid reply: '
                    f'Expected either "{affirmative_word}" or "{negative_word}" in the prompt: "{prompt_text}"'
                )
        # check if caption matches the setting
        expected_caption = affirmative_word if preset_boolean else negative_word
        expected_cp_caption = negative_word if preset_boolean else affirmative_word
        for caption_text, expected_caption_text, label in [
            (caption, expected_caption, 'main'), (cp_caption, expected_cp_caption, 'contrapositive')
        ]:
            if caption_text != expected_caption_text:
                raise ValueError(
                    f'Possibly Invalid reply: '
                    f'Expected caption "{expected_caption_text}" for the {label} proposition (preset: {preset_boolean}), '
                    f'found "{caption_text}"'
                )

        # check if both propositions are identical
        if prompt.strip().lower() == cp_prompt.strip().lower():
            raise ValueError(
                f'Possibly Invalid reply: contrapositive proposition ("{prompt}") '
                f'is identical to the main proposition ("{cp_prompt}")'
            )
        return True
