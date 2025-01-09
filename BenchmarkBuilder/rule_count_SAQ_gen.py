import os
import random
import sys

import click
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.rule_based import RuleBasedQuestionGenerator
from utils.io import enum_files, confirm_overwrite_file

PROMPT_QUESTION_TEMPLATE = [
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

PROMPT_HINT_TEMPLATE = [
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


class CountQGenerator(RuleBasedQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            exclude_labels: list[str] = None,
    ):
        super().__init__(
            scene_stat_json_file,
            export_json_file='./output/NUM_count_Q.json',
        )
        self.exclude_labels = exclude_labels or ['wall', 'floor', 'ceiling', 'object']

    def _generate_question(self) -> dict[str, str | int | float]:
        """Generate a counting question based on the scene data"""
        valid_object_labels = [
            label for label in self.scene_data.unique_labels
            if label not in self.exclude_labels
        ]
        # randomly select an object label
        object_name = random.choice(valid_object_labels)

        # count the number of instances of the object
        object_count = len(self.scene_data.get_instances_by_label(object_name))

        prompt_question = random.choice(PROMPT_QUESTION_TEMPLATE).replace('<OBJ>', object_name)
        prompt_hint = random.choice(PROMPT_HINT_TEMPLATE)

        return {
            'scene_id': self.scene_data.scene_id,
            'prompt': prompt_question + prompt_hint,
            'caption': str(object_count),
            'CoT_caption': f'<<answer:{object_count}>>',
            'obj_label': object_name,
        }


@click.command()
@click.option('--scene_stat_file',
              type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
              prompt='Enter the scene statistics file path',
              help='The directory or file containing the scene statistics in JSON format')
@click.option('--export_json_file',
              type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
              default='./output/NUM_count_Q.json',
              help='The file to which the generated questions will be exported')
@click.option('--n_questions', default=5, type=click.IntRange(1, None),
              help='Number of questions to generate')
@click.option('-s', '--skip_confirm', is_flag=False,
              help='Skip the confirmation prompt before processing the question JSONs')
def cli(
        scene_stat_file: str,
        export_json_file: str,
        n_questions: int,
        skip_confirm: bool,
):
    """CLI for generating the counting question"""
    scene_stat_files = enum_files(scene_stat_file, '.json', skip_confirm)

    # confirm overwrite existing file
    print(f'Exporting the counting questions to: {export_json_file}')
    if not confirm_overwrite_file(export_json_file):
        raise Exception(f'Aborted overwriting the existing file: {export_json_file}')

    print(f'Expected to generate {n_questions * len(scene_stat_files)} '
          f'(={n_questions} questions x {len(scene_stat_files)} scenes in total)')
    print(f'{f" Started generating counting questions ":=^80}')
    for scene_stat_file in tqdm(scene_stat_files, desc='Processing', unit='scene'):
        count_generator = CountQGenerator(scene_stat_json_file=scene_stat_file)
        count_generator.generate(n_questions)

    print(f'{f" Finished generating counting questions":=^80}')


if __name__ == '__main__':
    cli()
