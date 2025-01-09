import os
import random
import re
import string
import sys
from typing import Literal

import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.LLM_base import LLMBasedQuestionGenerator
from utils.io import parse_json_text, enum_files


class MCQGenerator(LLMBasedQuestionGenerator):
    def __init__(
            self,
            question_json_file: str,
            llm_model: str = 'qwen2.5:72b',
            llm_backend: Literal['ollama', 'openai'] = 'ollama',
            export_dir: str = './output/',
            n_options: int = 3,
            is_evenly_shuffled_options: bool = False,
    ):
        super().__init__(
            question_json_file=os.path.abspath(question_json_file),
            rewrite_question_type='MCQ',
            llm_model=llm_model,
            llm_backend=llm_backend,
            export_dir=export_dir,
        )

        # number of options for multiple-choice questions
        self.n_options = n_options
        # whether to strictly distribute the options evenly across the answer space
        self.is_evenly_shuffled_options = is_evenly_shuffled_options
        # if n_options is less than 2, raise an error
        if n_options < 2:
            raise ValueError('Number of options must be at least 2')

    def _rewrite_question(
            self,
            src_question: str,
            src_answer: str,
            question_set_idx: int,
            **kwargs
    ) -> dict[str, str | int | float]:
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
            f'Please rewrite the provided question into a multiple-choice format '
            f'with 1 correct answer and {int(self.n_options - 1)} alternative incorrect answers. '
            f'The main body of the question should NOT include the full text of the correct answer. '
            f'Ensure that the rewrite question, options and answer and logical and concise. '
            f'The correct answer should be placed in the provided correct option label. \n'
            # --- instructions for fixing mismatched questions and answers ---
            # f'In some worse case that the provided question and answer do not match concisely, '
            # f'try to modify the question in the minimum way possible to make them match. '
            # f'The updated question should reply to the same answer, and still logically and grammatically correct. '
            # f'NEVER modify the answer to fit the question. \n'
            # ----------------------------------------------------------------
            f'When generating alternative options, ensure they are: \n'
            f'- Of the same type (e.g., if the correct answer is a noun, the alternatives should also be nouns). \n'
            f'- Distinguishable from each other by object or entity, rather than just using synonyms. \n'
            f'- Plausible and realistic to avoid making it too easy to guess the correct answer. \n'
            f'The correct answer should be placed in the provided correct option label. '
            f'Separate the question from the options with a space and format the options as "A) option_a B) option_b", etc. '
            f'Reply only with a JSON object containing the following fields: \n'
            f'"question" - includes the multiple-choice question followed by the formatted options. \n'
            f'"answer" - contains the letter corresponding to the correct option (e.g., "A"). \n'
            f'Do not include any additional instructions or explanations other than what is described above. \n'
            f'The correct option should be {preset_rewritten_option}. \n'
            f'Question: "{src_question}".\nCorrect answer: "{src_answer_cleanup}".',
        )
        # get the rewritten question from the LLM model
        rewritten_question_dict = parse_json_text(
            self._chat_with_llm(llm_prompt), ['question', 'answer'])

        return {
            'src_prompt': src_question,
            'src_caption': src_answer,
            'prompt': rewritten_question_dict['question'],
            'caption': preset_rewritten_option,
            'CoT_caption': f'<<answer:{preset_rewritten_option}>>',
            'question_set_idx': question_set_idx,
            'question_type': 'MultipleChoice',
        }

    def _validate_rewritten_question(
            self,
            rewrite_output_dict: dict[str, str | int | float]
    ) -> bool:
        """Validate the rewritten multiple-choice question"""
        options_pattern = re.compile(r'([A-Z])\)\s*(.*?)\s*(?=[A-Z]\)|$)')
        mcq_options = {k: v for k, v in options_pattern.findall(rewrite_output_dict['prompt'])}
        # check if the number of options matches the preset
        if (not mcq_options) or (len(mcq_options) != self.n_options):
            raise ValueError(f'Possibly Invalid reply: '
                             f'expected {self.n_options} options, found {len(mcq_options)}: '
                             f'{mcq_options} in "{rewrite_output_dict["prompt"]}"')
        # check if the correct answer is at the provided correct option label
        src_answer = rewrite_output_dict['src_caption'].lower().rstrip('.').strip()
        preset_rewritten_option = rewrite_output_dict['caption']
        rewrite_answer = mcq_options.get(preset_rewritten_option, '').lower().rstrip('.').strip()
        if src_answer != rewrite_answer:
            raise ValueError(f'Possibly Invalid reply: '
                             f'correct answer "{src_answer}" not found in {preset_rewritten_option}) of "{mcq_options}"')

        # check if the full text of the correct answer is not included in the main body of the question
        if rewrite_output_dict['src_prompt'].lower() in rewrite_output_dict['prompt'].lower():
            raise ValueError(f'Possibly Invalid reply: '
                             f'full text of the correct answer "{rewrite_output_dict["src_prompt"]}" '
                             f'found in the main body of the question: "{rewrite_output_dict["prompt"]}"')

        return True


@click.command()
@click.option('--llm_model', default='qwen2.5:72b', type=str,
              help='The LLM model used for rewriting the question_json')
@click.option('--llm_backend', default='ollama', type=click.Choice(['ollama', 'openai']),
              help='The backend service for the LLM model. Choose between Ollama (default) or OpenAI')
@click.option('--question_file', type=click.Path(readable=True, dir_okay=True),
              prompt='Enter the input directory or file path',
              help='The directory or file containing the question JSONs')
@click.option('--n_option', default=3, type=click.IntRange(2, None),
              help='Number of options for the multiple-choice question')
@click.option('--evenly_shuffled', is_flag=True,
              help='Distribute the options evenly across the answer space')
@click.option('--max_retry', default=5, type=click.IntRange(1, None),
              help='Maximum number of retries for each question when failed to rewrite')
@click.option('--export_dir', default='./output/',
              type=click.Path(file_okay=False, writable=True),
              help='The directory to export the rewritten question JSONs')
@click.option('-s', '--skip_confirm', is_flag=True,
              help='Skip the confirmation prompt before processing the question JSONs')
def cli(
        llm_model: str,
        llm_backend: Literal['ollama', 'openai'],
        question_file: str,
        n_option: int,
        evenly_shuffled: bool,
        max_retry: int,
        export_dir: str,
        skip_confirm: bool = False
):
    """CLI for rewriting the question_json as multiple-choice question_json"""
    question_jsons = enum_files(question_file, '.json', skip_confirm)

    print(f'{f" Start rewriting {len(question_jsons)} question JSON files ":=^80}')
    for question_json in question_jsons:
        mcq_generator = MCQGenerator(
            question_json_file=question_json,
            llm_model=llm_model,
            llm_backend=llm_backend,
            export_dir=export_dir,
            n_options=n_option,
            is_evenly_shuffled_options=evenly_shuffled
        )
        mcq_generator.rewrite(max_retries=max_retry)
        print(f'Rewritten questions exported to: {mcq_generator.export_dir}')

    print(f'{f" Finished rewriting {len(question_jsons)} question JSON files":=^80}')


if __name__ == '__main__':
    cli()
