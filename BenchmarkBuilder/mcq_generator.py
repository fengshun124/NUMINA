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
        f'Ensure the revised question remains concise and aligned with the original query. '
        f'The main body of the question should NOT include the full text of the correct answer. '
        f'Please make sure that the question reflects clearly the information of the original correct answer. '
        f'When generating alternative options, ensure they are: \n'
        f'- Of the same type (e.g., if the correct answer is a noun, the alternatives should also be nouns). \n'
        f'- Distinguishable from each other by object or entity, rather than just using synonyms. \n'
        f'- Plausible and realistic to avoid making it too easy to guess the correct answer. \n'
        f'The correct answer should be placed in the provided correct option label. '
        f'Separate the question from the options with a space, '
        f'and format the options as "A) option_a B) option_b", etc. '
        f'Please reply only with a JSON object containing "question" and "answer" fields, '
        f'without any additional text or explanations. \n'
        f'"question" should include the multiple-choice question followed by the formatted options. \n'
        f'"answer" should contain only the letter corresponding to the correct option (e.g., "A"). \n'
        f'Ensure that the rewrite text is clear and concise, maintaining the original meaning of the question. \n'
        f'The correct option should be {preset_answer_label}. \n'
        f'Question: "{question_text}".\nCorrect answer: "{cleanup_correct_answer}".',
    )
    response = chat_llm(llm_prompt, **kwargs)

    # validate if response can be parsed as a JSON object
    response_dict = parse_llm_response(response, ['question', 'answer'])
    mcq_text = response_dict['question'].strip()
    mcq_correct_answer_label = response_dict['answer'].strip().upper()

    # validate if the correct option is consistent with the preset correct option
    if mcq_correct_answer_label != preset_answer_label:
        raise ValueError(f'Possible invalid reply: "{mcq_correct_answer_label}" from '
                         f'"{mcq_text}" does not match the preset correct answer "{preset_answer_label}"')

    option_pattern = re.compile(r'[A-Z]\)\s*(.*?)\s*(?=\s*[A-Z]\)|$)')
    mcq_options = option_pattern.findall(mcq_text)
    # validate if exactly given number of options are generated
    if len(mcq_options) != n_option:
        raise ValueError(f'Possible invalid reply: expect exactly {n_option} options from '
                         f'"{mcq_text}". Extracted {len(mcq_options)} options: {mcq_options}')
    # validate if the correct option is consistent with the preset correct option
    if cleanup_correct_answer.lower() not in [option.lower() for option in mcq_options]:
        raise ValueError(f'Possible invalid reply: '
                         f'{mcq_options} from "{mcq_text}" does not contain "{cleanup_correct_answer}"')

    mcq_main_text = option_pattern.sub('', mcq_text).strip()
    if correct_answer.lower() in mcq_main_text.lower():
        raise ValueError(f'Possible invalid reply: '
                         f'"{mcq_main_text}" from "{mcq_text}" contains "{cleanup_correct_answer}"')

    return {
        'prompt': mcq_text,
        'caption': mcq_correct_answer_label,
        'src_prompt': question_text,
        'src_caption': correct_answer,
    }


@click.command()
@click.option('--llm_model', default='qwen2.5:72b', type=str)
@click.option('--llm_backend', default='ollama', type=click.Choice(['ollama', 'openai']))
@click.option('--questions', type=click.Path(readable=True, dir_okay=True),
              prompt='Enter the input directory or file path')
@click.option('--n_option', default=3, type=click.IntRange(2, None))
@click.option('--max_retry', default=5, type=click.IntRange(1, None))
@click.option('--export_dir', default='./output/',
              type=click.Path(file_okay=False, writable=True))
@click.option('--export_prefix', default='MCQ', type=str)
@click.option('-s', '--skip_confirm', is_flag=True)
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
