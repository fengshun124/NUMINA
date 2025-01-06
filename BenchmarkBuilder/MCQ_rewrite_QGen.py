import random
import re
import string

import click

from utils.chat import chat_llm
from utils.misc import process_question_jsons, parse_llm_response


def generate_multiple_choice_question(
        question_text: str, correct_answer: str,
        n_option: int = 3, preset_answer_label: str | None = None,
        **kwargs
) -> dict:
    """Generate multiple-choice question based on the given question and correct option via LLM"""
    if n_option < 2:
        raise ValueError('question must have at least 2 options')
    if type(n_option) != int:
        raise ValueError('n_option must be an integer')

    # cleanup the correct option string
    cleanup_correct_answer = correct_answer.rstrip('.').strip()
    # if correct option label is not provided, randomly select one
    preset_answer_label = preset_answer_label or random.choice(string.ascii_uppercase[:n_option])

    # generate the multiple-choice question via LLM
    llm_prompt = str(
        f'Please rewrite the provided question into a multiple-choice format '
        f'with 1 correct answer and {int(n_option - 1)} alternative incorrect answers. '
        f'The main body of the question should NOT include the full text of the correct answer. '
        f'Ensure that the rewrite question, options and answer and logical and concise. '
        f'The correct answer should be placed in the provided correct option label. '
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
        f'The correct option should be {preset_answer_label}. \n'
        f'Question: "{question_text}".\nCorrect answer: "{cleanup_correct_answer}".',
    )
    response = chat_llm(llm_prompt, **kwargs)

    # validate if response can be parsed as a JSON object
    response_dict = parse_llm_response(response, ['question', 'answer'])
    mcq_full_text = response_dict['question'].strip()
    mcq_answer_label = response_dict['answer'].strip()

    if mcq_answer_label.lower() != preset_answer_label.lower():
        raise ValueError(f'Possibly Invalid reply: '
                         f'"{mcq_answer_label}" does not match the preset correct answer "{preset_answer_label}"')

    options_pattern = re.compile(r'([A-Z])\)\s*(.*?)\s*(?=[A-Z]\)|$)')
    mcq_options = {k: v for k, v in options_pattern.findall(mcq_full_text)}
    # validate if exactly given number of options are generated
    if (not mcq_options) or (len(mcq_options) != n_option):
        raise ValueError(f'Possibly Invalid reply: '
                         f'expected {n_option} options, found {len(mcq_options)}: {mcq_options} in "{mcq_full_text}"')

    # validate if the correct option is present in the options
    if cleanup_correct_answer.lower() != mcq_options.get(preset_answer_label).lower().strip('.'):
        raise ValueError(f'Possibly Invalid reply: '
                         f'correct answer "{cleanup_correct_answer}" not found in the options: {mcq_options}')

    # validate if the question text does NOT contain the full text of the correct answer
    mcq_question_text = options_pattern.sub('', mcq_full_text).strip()
    if cleanup_correct_answer.lower() in mcq_question_text.lower():
        raise ValueError(f'Possibly Invalid reply: '
                         f'{mcq_question_text} contains the full text of the correct answer "{cleanup_correct_answer}"')

    return {
        'prompt': mcq_full_text,
        'caption': mcq_answer_label,
        'CoT_caption': f'{{"answer": "{mcq_answer_label}"}}',
        'question_type': 'MultipleChoice',
        'src_prompt': question_text,
        'src_caption': correct_answer,
    }


@click.command()
@click.option('--llm_model', default='qwen2.5:72b', type=str,
              help='The LLM model used for rewriting the questions')
@click.option('--llm_backend', default='ollama', type=click.Choice(['ollama', 'openai']),
              help='The backend service for the LLM model. Choose between Ollama (default) or OpenAI')
@click.option('--questions', type=click.Path(readable=True, dir_okay=True),
              prompt='Enter the input directory or file path',
              help='The directory or file containing the question JSONs')
@click.option('--n_option', default=3, type=click.IntRange(2, None),
              help='Number of options for the multiple-choice question')
@click.option('--max_retry', default=5, type=click.IntRange(1, None),
              help='Maximum number of retries for each question when failed to rewrite')
@click.option('--export_dir', default='./output/',
              type=click.Path(file_okay=False, writable=True),
              help='The directory to export the rewritten question JSONs')
@click.option('--export_prefix', default='MCQ', type=str,
              help='The prefix for the exported rewritten question JSONs. Default is "MCQ" '
                   '(e.g., question_A.json -> MCQ-question_A.json)')
@click.option('-s', '--skip_confirm', is_flag=True,
              help='Skip the confirmation prompt before processing the question JSONs')
def main(
        llm_model, llm_backend,
        questions,
        n_option, max_retry,
        export_dir, export_prefix,
        skip_confirm
):
    process_question_jsons(
        question_json_path=questions,
        rewrite_type='MCQ', rewrite_method=generate_multiple_choice_question,
        export_dir=export_dir, export_prefix=export_prefix,
        skip_confirm=skip_confirm,
        max_retry=max_retry,
        llm_model=llm_model, llm_backend=llm_backend,
        n_option=n_option
    )


if __name__ == '__main__':
    main()
