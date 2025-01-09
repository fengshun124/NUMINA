import os
import random
import sys

import click
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.rule_based import RuleBasedQuestionGenerator
from utils.io import enum_files, confirm_overwrite_file

PROMPT_QUESTION_TEMPLATES = [
    'Can you count the number of <OBJ> in the room? ',
    'Can you count the <OBJ> in the room? ',
    'Can you tell me how many <OBJ> there are in the room? ',
    'Can you tell me the number of <OBJ> in the room? ',
    'Count the number of <OBJ> in the room. ',
    'How many <OBJ> can you see in the room? ',
    'How many <OBJ> are there in the room? ',
    'How many <OBJ> are in the room? ',
    'Please count the number of <OBJ> in the room. ',
    'Please provide the number of <OBJ> in the room. ',
    'Please tell me the number of <OBJ> in the room. ',
    'What is the number of <OBJ> in the room? ',
    'What is the count of <OBJ> in the room? ',
]

PROMPT_HINT_TEMPLATES = [
    'Kindly provide a number as the answer. ',
    'Give a number as the answer. ',
    'Give a numerical response. ',
    'Offer a number as the answer. ',
    'Provide a number as the answer. ',
    'Please provide a numerical response. ',
    'Please provide a number as the answer. ',
    'Please reply with a number only. ',
    'Reply with a number only. ',
]


class CountSAQGenerator(RuleBasedQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str = './output/NUM-count-Q.json',
            exclude_labels: list[str] = None,
    ):
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            exclude_labels,
        )

    def generate(
            self,
            n_questions: int,
            max_retries: int = 20,
            is_strict_no_reuse: bool = False,
    ) -> None:
        """Generate a counting question based on the scene data"""
        self._validate_args(n_questions, max_retries)

        valid_object_labels = [
            label for label in self.scene_data.unique_labels
            if label not in self.exclude_labels
        ]

        question_dicts = []
        if len(valid_object_labels) < n_questions:
            if is_strict_no_reuse:
                print(f'[Warning] Only generating {len(valid_object_labels)} questions '
                      f'instead of requested {n_questions} to avoid label reuse')
                n_questions = len(valid_object_labels)
            else:
                print(f'[Warning] There are fewer unique object labels ({len(valid_object_labels)}) '
                      f'than the number of questions ({n_questions}). Some questions may refer to the same object.')

        used_obj_labels, remaining_obj_labels = [], valid_object_labels.copy()

        while len(question_dicts) < n_questions and max_retries > 0:
            if not remaining_obj_labels:
                if is_strict_no_reuse:
                    break
                remaining_obj_labels = used_obj_labels.copy()
                used_obj_labels.clear()
                print(f'[Warning] Reusing object labels for the questions')

            obj_label = random.choice(remaining_obj_labels)
            remaining_obj_labels.remove(obj_label)
            used_obj_labels.append(obj_label)

            question_dict = self._form_question(obj_label)
            if question_dict not in question_dicts:
                question_dicts.append(question_dict)
                self._export_question(question_dict)

            max_retries -= 1

        if len(question_dicts) < n_questions:
            print(f'[Warning] Could only generate {len(question_dicts)} unique questions '
                  f'instead of the requested {n_questions}')

    def _form_question(self, obj_label: str) -> dict[str, str]:
        """Form the question set based on the object label"""
        object_count = len(self.scene_data.get_instances_by_label(obj_label))

        prompt_question = random.choice(PROMPT_QUESTION_TEMPLATES).replace('<OBJ>', obj_label)
        prompt_hint = random.choice(PROMPT_HINT_TEMPLATES)

        return {
            'scene_id': self.scene_data.scene_id,
            'obj_label': obj_label,
            'prompt': prompt_question + prompt_hint,
            'caption': str(object_count),
            'CoT_caption': f'<<answer:{object_count}>>',
        }

    @staticmethod
    def _validate_args(n_questions: int, max_retries: int, ) -> None:
        if n_questions < 1:
            raise ValueError('Number of questions must be at least 1')

        if max_retries < n_questions:
            raise Exception(f'Maximum number of retries ({max_retries}) must be '
                            f'no less than the number of questions ({n_questions})')


@click.command()
@click.option('--scene_stat_file',
              type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
              prompt='Enter the scene statistics file path',
              help='The directory or file containing the scene statistics in JSON format')
@click.option('--export_json_file',
              type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
              default='./output/NUM-count-SAQ.json',
              help='The file to which the generated questions will be exported')
@click.option('--n_questions', default=5, type=click.IntRange(1, None),
              help='Number of questions to generate')
@click.option('--exclude_labels', default='wall,floor,ceiling,object', type=str,
              help='Comma-separated list of labels to exclude from the question generation. '
                   'Default: wall, floor, ceiling, object. '
                   '(e.g., --exclude_labels=wall,floor,ceiling,object)')
@click.option('--strict_no_reuse', is_flag=True, default=False,
              help='Stop generating questions when all labels are used instead of reusing them')
@click.option('-s', '--skip_confirm', is_flag=False,
              help='Skip the confirmation prompt before processing the question JSONs')
def cli(
        scene_stat_file: str,
        export_json_file: str,
        n_questions: int,
        exclude_labels: str,
        skip_confirm: bool,
        strict_no_reuse: bool,
):
    """CLI for generating the counting question"""
    scene_stat_files = enum_files(scene_stat_file, '.json', skip_confirm)

    # validate export file, overwrite it if exists and confirmed
    print(f'Exporting the counting questions to: {export_json_file}')
    if not confirm_overwrite_file(export_json_file):
        raise Exception(f'Aborted overwriting the existing file: {export_json_file}')

    print(f'Expected to generate {n_questions * len(scene_stat_files)} '
          f'(={n_questions} questions x {len(scene_stat_files)} scenes in total)')

    # exclude the specified labels from the question generation
    exclude_labels = [label.strip() for label in exclude_labels.split(',')]
    print(f'Excluding objects with labels: {exclude_labels} while generating the questions')

    print(f'{f" Started generating counting questions ":=^80}')
    for scene_stat_file in tqdm(scene_stat_files, desc='Processing', unit='scene'):
        count_generator = CountSAQGenerator(
            scene_stat_json_file=scene_stat_file,
            export_json_file=export_json_file,
        )
        count_generator.generate(
            n_questions=n_questions,
            is_strict_no_reuse=strict_no_reuse
        )

    print(f'{f" Finished generating counting questions ":=^80}')


if __name__ == '__main__':
    cli()
