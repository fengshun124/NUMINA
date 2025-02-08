import os
import random
import sys
from typing import Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BenchmarkBuilder.LLM.base import LLMBasedQuestionGenerator
from BenchmarkBuilder.utils.io import parse_json_text


class TFQRewriter(LLMBasedQuestionGenerator):
    """True/False question generator implementation"""

    def __init__(self, **kwargs):
        # extract TFQ-specific parameters before calling parent init
        self.is_evenly_shuffled_boolean = kwargs.pop('is_evenly_shuffled_boolean', False)

        # Call parent init with remaining kwargs
        super().__init__(
            question_json_file=kwargs['question_json_file'],
            rewrite_question_type='TFQ',
            llm_model=kwargs.get('llm_model', 'qwen2.5:72b'),
            llm_backend=kwargs.get('llm_backend', 'ollama'),
            output_path=kwargs.get('export_dir', './output/')
        )

    def _rewrite_question(
            self,
            src_question: str,
            src_answer: str,
            question_set_idx: int,
            **kwargs
    ) -> dict[str, str | int | float]:
        """Rewrite the question as a true/false question"""
        # cleanup answer text
        src_answer_cleanup = src_answer.rstrip('.').strip()
        # generate rewrite caption
        if self.is_evenly_shuffled_boolean:
            preset_rewritten_boolean = True if question_set_idx % 2 == 0 else False
        else:
            preset_rewritten_boolean = random.choice([True, False])

        affirmative_words = ['Yes', 'True', 'Correct', 'Agree']
        negative_words = ['No', 'False', 'Incorrect', 'Disagree']
        # choose from pairs of words
        affirmative_word, negative_word = random.choice([*zip(affirmative_words, negative_words)])

        # generate the LLM prompt
        llm_prompt = str(
            f'Please rewrite the provided question into a True/False question '
            f'using {affirmative_word} and {negative_word} as the answer options. '
            f'The revised question should use concise and logical expression, illustrating the original query. '
            f'Question: "{src_question}".\n Correct answer: "{src_answer_cleanup}".\n'
        )
        if preset_rewritten_boolean:
            llm_prompt += str(
                f'The correct answer is {affirmative_word}. '
                f'As you rewrite the question, ensure that the correct answer should appear exactly in the question. '
            )
        else:
            llm_prompt += str(
                f'The correct answer is {negative_word}. Please generate an alternative option, ensuring it is: \n'
                f'- Of the same type (e.g., if the correct answer is a noun, the alternatives should also be nouns). \n'
                f'- Distinguishable from each other by object or entity, rather than just using synonyms. '
                f'As you rewrite the question, ensure that the correct answer should NOT appear in the question. '
            )
        llm_prompt += str(
            'Please reply only with the revised question with answer hint, '
            f'as well as "{affirmative_word}" or "{negative_word}" '
            f'with separate spaces and without any additional text or explanations. For instance: '
            f'{{"question": "Alice sits next to Bob. Is that statement correct? Answer with Yes or No", "answer": "No"}}'
        )
        # get the rewritten question from the LLM model
        rewritten_question_dict = parse_json_text(
            self._chat_with_llm(llm_prompt), ['question', 'answer'])

        return {
            'src_prompt': src_question,
            'src_caption': src_answer,
            'prompt': rewritten_question_dict['question'],
            'caption': affirmative_word if preset_rewritten_boolean else negative_word,
            'CoT_caption': f'<<answer:{affirmative_word if preset_rewritten_boolean else negative_word}>>',
            'ref_captions': affirmative_words if preset_rewritten_boolean else negative_words,
            'boolean_word_pairs': f'{affirmative_word} / {negative_word}',
            'preset_answer': preset_rewritten_boolean,
            'question_set_idx': question_set_idx,
            'question_type': 'LLM_rewrite-TFQ',
            'llm': {
                'model': self.llm_model,
                'backend': self.llm_backend
            }
        }

    def _validate_rewritten_question(
            self,
            rewrite_output_dict: dict[str, Union[str, int, float]]
    ) -> bool:
        """Validate the rewritten true/false question"""
        boolean_word_pairs = rewrite_output_dict['boolean_word_pairs'].split(' / ')
        # check if main body of the question contains the desired affirmative/negative words
        if not all(word in rewrite_output_dict['prompt'] for word in boolean_word_pairs):
            raise ValueError(f'Possibly Invalid reply: '
                             f'descriptive words "{boolean_word_pairs}" not found in the main body of the question: ')

        # check if the rewrite answer matches the preset answer
        if (rewrite_output_dict['preset_answer'] and
            boolean_word_pairs[0] != rewrite_output_dict['caption']) or (
                not rewrite_output_dict['preset_answer'] and
                boolean_word_pairs[1] != rewrite_output_dict['caption']):
            raise ValueError(
                f'Possibly Invalid reply: preset answer "{boolean_word_pairs[0]}" '
                f'does not match the rewrite answer: "{rewrite_output_dict["caption"]}"'
            )

        return True
