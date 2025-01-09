import os
import random
import sys
from typing import Literal

import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.LLM_base import LLMBasedQuestionGenerator
from utils.io import parse_json_text, enum_files


class TFQGenerator(LLMBasedQuestionGenerator):
    def __init__(
            self,
            question_json_file: str,
            llm_model: str = 'qwen2.5:72b',
            llm_backend: Literal['ollama', 'openai'] = 'ollama',
            export_dir: str = './output/',
            is_evenly_shuffled_boolean: bool = False,
    ):
        super().__init__(
            question_json_file=os.path.abspath(question_json_file),
            rewrite_question_type='TFQ',
            llm_model=llm_model,
            llm_backend=llm_backend,
            export_dir=export_dir
        )

        # whether to strictly distribute the options evenly across the answer space
        self.is_evenly_shuffled_options = is_evenly_shuffled_boolean

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
        if self.is_evenly_shuffled_options:
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
            'boolean_word_pairs': f'{affirmative_word} / {negative_word}',
            'preset_answer': preset_rewritten_boolean,
            'question_set_idx': question_set_idx,
            'question_type': 'TrueFalse',
        }

    def _validate_rewritten_question(
            self,
            rewrite_output_dict: dict[str, str | int | float]
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


@click.command()
@click.option('--llm_model', default='qwen2.5:72b', type=str,
              help='The LLM model used for rewriting the question_json')
@click.option('--llm_backend', default='ollama', type=click.Choice(['ollama', 'openai']),
              help='The backend service for the LLM model. Choose between Ollama (default) or OpenAI')
@click.option('--question_file', type=click.Path(readable=True, dir_okay=True),
              prompt='Enter the input directory or file path',
              help='The directory or file containing the question JSONs')
@click.option('--evenly_shuffled', is_flag=True,
              help='Distribute the options evenly across the answer space')
@click.option('--max_retry', default=5, type=click.IntRange(1, None),
              help='Maximum number of retries for each question when failed to rewrite')
@click.option('--export_dir', default='./output/',
              type=click.Path(file_okay=False, writable=True),
              help='The directory to export the rewritten question JSONs')
@click.option('-s', '--skip_confirm', is_flag=False,
              help='Skip the confirmation prompt before processing the question JSONs')
def cli(
        llm_model: str,
        llm_backend: Literal['ollama', 'openai'],
        question_file: str,
        evenly_shuffled: bool,
        max_retry: int,
        export_dir: str,
        skip_confirm: bool
):
    """CLI for rewriting the questions as True/False questions"""
    question_jsons = enum_files(question_file, '.json', skip_confirm)

    print(f'{f" Start rewriting {len(question_jsons)} question JSON files ":=^80}')
    for question_json in question_jsons:
        mcq_generator = TFQGenerator(
            question_json_file=question_json,
            llm_model=llm_model,
            llm_backend=llm_backend,
            export_dir=export_dir,
            is_evenly_shuffled_boolean=evenly_shuffled
        )
        mcq_generator.rewrite(max_retries=max_retry)
        print(f'Rewritten questions exported to: {mcq_generator.export_dir}')

    print(f'{f" Finished rewriting {len(question_jsons)} question JSON files":=^80}')


if __name__ == '__main__':
    cli()
