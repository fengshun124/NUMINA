import random

import click

from utils.chat import chat_llm
from utils.misc import process_question_jsons, parse_llm_response


def generate_true_or_false_question(
        question_text: str, correct_answer: str,
        preset_answer_boolean: bool | None = None,
        **kwargs
) -> dict:
    """Generate True or False question based on the given question and correct option via LLM"""
    # cleanup the correct option string
    cleanup_correct_answer = correct_answer.rstrip('.').strip()
    # if correct option label is not provided, randomly select one
    preset_answer_boolean = preset_answer_boolean or random.choice([True, False])

    affirmative_words = ['Yes', 'True', 'Correct', 'Agree']
    negative_words = ['No', 'False', 'Incorrect', 'Disagree']
    # choose from pairs of words
    affirmative_word, negative_word = random.choice([*zip(affirmative_words, negative_words)])

    llm_prompt = str(
        f'Please rewrite the provided question into a True/False question '
        f'using {affirmative_word} and {negative_word} as the answer options. '
        f'The revised question should use concise and logical expression, illustrating the original query. '
        f'Question: "{question_text}".\n Correct answer: "{cleanup_correct_answer}".\n'
    )
    if preset_answer_boolean:
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
    response = chat_llm(llm_prompt, **kwargs)

    # check if response can be parsed as a JSON object
    response_dict = parse_llm_response(response, ['question', 'answer'])
    tfq_text = response_dict['question'].strip()
    tfq_answer = response_dict['answer'].strip()

    # validate if question contains the desired affirmative/negative words
    if not all(word.lower() in tfq_text.lower() for word in [affirmative_word.lower(), negative_word.lower()]):
        raise ValueError(f'Possible invalid reply: '
                         f'"{tfq_text}" does not contain the given affirmative/negative words'
                         f' {affirmative_word}, {negative_word}')

    # validate if the option aligns with the preset correct option
    if preset_answer_boolean and tfq_answer not in affirmative_words:
        raise ValueError(f'Possible invalid reply: '
                         f'"{tfq_answer}" is not in the given affirmative words {affirmative_words}')
    if not preset_answer_boolean and tfq_answer not in negative_words:
        raise ValueError(f'Possible invalid reply: '
                         f'"{tfq_answer}" is not in the given negative words {negative_words}')

    return {
        'prompt': tfq_text,
        'caption': tfq_answer,
        'question_type': 'TrueFalse',
        'preset_answer': preset_answer_boolean,
        'src_prompt': question_text,
        'src_answer': correct_answer,
    }


@click.command()
@click.option('--llm_model', default='qwen2.5:72b', type=str)
@click.option('--llm_backend', default='ollama', type=click.Choice(['ollama', 'openai']))
@click.option('--questions', type=click.Path(readable=True, dir_okay=True),
              prompt='Enter the input directory or file path')
@click.option('--max_retry', default=5, type=click.IntRange(1, None))
@click.option('--export_dir', default='./output/',
              type=click.Path(file_okay=False, writable=True))
@click.option('--export_prefix', default='TFQ', type=str)
@click.option('-s', '--skip_confirm', is_flag=True)
def main(
        llm_model, llm_backend,
        questions,
        max_retry,
        export_dir, export_prefix,
        skip_confirm
):
    process_question_jsons(
        question_json_path=questions,
        rewrite_type='TFQ', rewrite_method=generate_true_or_false_question,
        export_dir=export_dir, export_prefix=export_prefix,
        skip_confirm=skip_confirm,
        max_retry=max_retry,
        llm_model=llm_model, llm_backend=llm_backend
    )


if __name__ == '__main__':
    main()
