import os
import random
import sys
from itertools import combinations

import click
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.rule_based import RuleBasedQuestionGenerator
from utils.io import enum_files, confirm_overwrite_file

RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the distance between <OBJ1> and <OBJ2> greater than the distance between <OBJ3> and <OBJ4>? ',
            'Is <OBJ1> and <OBJ2> further than <OBJ3> and <OBJ4>? '
        ]
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the distance between <OBJ1> and <OBJ2> less than the distance between <OBJ3> and <OBJ4>? ',
            'Is <OBJ3> and <OBJ4> closer than <OBJ1> and <OBJ2>? '
        ]
    },
    '=': {
        'func': lambda x, y: abs(x - y) < 0.1,
        'text': 'approximately equal to',
        'templates': [
            'Is the distance between <OBJ1> and <OBJ2> approximately equal to the distance between <OBJ3> and <OBJ4>? ',
            'Is <OBJ1> and <OBJ2> about the same distance as <OBJ3> and <OBJ4>? '
        ]
    }
}

CoT_TEMPLATE = """The distance between <OBJ1> and <OBJ2> is <DIST1> meters.
The distance between <OBJ3> and <OBJ4> is <DIST2> meters.
Since the distance between <OBJ1> and <OBJ2> is <RELATION> the distance between <OBJ3> and <OBJ4>,
the correct answer is <<answer:<ANSWER>>>.
"""


class DistanceCompareTFQGenerator(RuleBasedQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str = './output/NUM-dist_compare-TFQ.json',
            exclude_labels: list[str] = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            exclude_labels,
        )

    def generate(
            self,
            n_questions: int = 2,
            max_retries: int = 20,
            is_strict_no_reuse: bool = False,
            is_evenly_shuffled_boolean: bool = False
    ) -> None:
        """Generate distance comparison True/False questions based on the scene data"""
        self._validate_args(n_questions, max_retries, is_evenly_shuffled_boolean)
        valid_pairwise_distance_dict = self._enum_valid_pairwise_distances()

        preset_booleans = [True, False] * (n_questions // 2) or random.choices([True, False], k=n_questions)
        question_dicts = []
        used_pairs, remaining_pairs = [], list(valid_pairwise_distance_dict.keys())

        while len(question_dicts) < n_questions and max_retries > 0:
            if not remaining_pairs:
                if is_strict_no_reuse:
                    break
                remaining_pairs = used_pairs.copy()
                used_pairs.clear()
                print(f'[Warning] Reusing object pairs for the questions')

            obj_pair1_keys = random.choice(remaining_pairs)
            remaining_pairs.remove(obj_pair1_keys)
            used_pairs.append(obj_pair1_keys)

            # choose a random relation
            relation = random.choice(list(RELATION_DICT.keys()))
            # construct the remaining set that satisfy the relation
            valid_remaining_pairs = [
                obj_pair2_keys for obj_pair2_keys in remaining_pairs
                if RELATION_DICT[relation]['func'](
                    valid_pairwise_distance_dict[obj_pair1_keys],
                    valid_pairwise_distance_dict[obj_pair2_keys]
                ) is preset_booleans[len(question_dicts)]
            ]
            if not valid_remaining_pairs:
                continue
            obj_pair2_keys = random.choice(valid_remaining_pairs)
            remaining_pairs.remove(obj_pair2_keys)

            question_dict = self._form_question(
                obj_pair1_keys,
                obj_pair2_keys,
                relation, preset_booleans[len(question_dicts)]
            )

            if question_dict not in question_dicts:
                question_dicts.append(question_dict)
                self._export_question(question_dict)

            max_retries -= 1

        if len(question_dicts) < n_questions:
            print(f'[Warning] Could only generate {len(question_dicts)} unique questions '
                  f'instead of the requested {n_questions}')

    def _form_question(
            self,
            obj_pair1_keys: str,
            obj_pair2_keys: str,
            relation: str,
            preset_boolean: bool
    ) -> dict[str, str]:
        """Form a distance comparison question template based on the object pairs"""
        obj1_id, obj2_id = obj_pair1_keys.split('-')
        obj3_id, obj4_id = obj_pair2_keys.split('-')

        obj1_label = self.scene_data.get_instance_by_object_id(obj1_id).label
        obj2_label = self.scene_data.get_instance_by_object_id(obj2_id).label
        obj3_label = self.scene_data.get_instance_by_object_id(obj3_id).label
        obj4_label = self.scene_data.get_instance_by_object_id(obj4_id).label

        return {
            'scene_id': self.scene_data.scene_id,
            'pairwise_meta': {
                'obj1': {'object_id': obj1_id, 'label': obj1_label},
                'obj2': {'object_id': obj2_id, 'label': obj2_label},
                'obj3': {'object_id': obj3_id, 'label': obj3_label},
                'obj4': {'object_id': obj4_id, 'label': obj4_label},
                'obj_pair1_key': obj_pair1_keys,
                'obj_pair1_distance': self.scene_data.get_pairwise_distance(obj1_id, obj2_id),
                'obj_pair2_key': obj_pair2_keys,
                'obj_pair2_distance': self.scene_data.get_pairwise_distance(obj3_id, obj4_id),
            },
            'prompt': random.choice(RELATION_DICT[relation]['templates']).replace(
                '<OBJ1>', obj1_label).replace(
                '<OBJ2>', obj2_label).replace(
                '<OBJ3>', obj3_label).replace(
                '<OBJ4>', obj4_label).replace(
                '<RELATION>', RELATION_DICT[relation]['text']
            ),
            'caption': str(preset_boolean),
            'CoT_caption': CoT_TEMPLATE.replace(
                '<OBJ1>', obj1_label).replace(
                '<OBJ2>', obj2_label).replace(
                '<OBJ3>', obj3_label).replace(
                '<OBJ4>', obj4_label).replace(
                '<DIST1>', f'{self.scene_data.get_pairwise_distance(obj1_id, obj2_id):>.1f}').replace(
                '<DIST2>', f'{self.scene_data.get_pairwise_distance(obj3_id, obj4_id):>.1f}').replace(
                '<RELATION>', RELATION_DICT[relation]['text']).replace(
                '<ANSWER>', str(preset_boolean)).replace(
                # line break for CoT readability
                '\n', ' '
            ),
        }

    def _enum_valid_pairwise_distances(self) -> dict[str, float]:
        """Enumerate the valid pairwise distances between objects"""
        sole_obj_ids = [
            str(inst.object_id) for inst in self.scene_data.instances
            if inst.label not in self.exclude_labels and
               len(self.scene_data.get_instances_by_label(inst.label)) == 1
        ]
        valid_pairwise_distances = {
            '-'.join([obj_id1, obj_id2]): self.scene_data.get_pairwise_distance(obj_id1, obj_id2)
            for obj_id1, obj_id2 in combinations(sole_obj_ids, 2)
        }
        return valid_pairwise_distances

    @staticmethod
    def _validate_args(
            n_questions: int,
            max_retries: int,
            is_evenly_shuffled: bool,
    ) -> None:
        if n_questions < 1:
            raise ValueError('Number of questions must be at least 1')
        if is_evenly_shuffled and n_questions % 2 != 0:
            raise ValueError('Even distribution requires even number of questions')
        if max_retries < n_questions:
            raise ValueError('max_retries must be >= n_questions')


@click.command()
@click.option('--scene_stat_file',
              type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
              prompt='Enter the scene statistics file path',
              help='The directory or file containing the scene statistics in JSON format')
@click.option('--export_json_file',
              type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
              default='./output/NUM-dist_compare-TFQ.json',
              help='The file to which the generated questions will be exported')
@click.option('--n_questions', default=2, type=click.IntRange(1, None),
              help='Number of questions to generate per scene')
@click.option('--exclude_labels', default='wall,floor,ceiling,object', type=str,
              help='Comma-separated list of labels to exclude from question generation')
@click.option('--strict_no_reuse', is_flag=True, default=False,
              help='Stop generating questions when all pairs are used instead of reusing them')
@click.option('--evenly_shuffled', is_flag=True, default=False,
              help='Distribute True/False answers evenly (requires even number of questions)')
@click.option('-s', '--skip_confirm', is_flag=True,
              help='Skip confirmation prompt before processing')
def cli(
        scene_stat_file: str,
        export_json_file: str,
        n_questions: int,
        exclude_labels: str,
        strict_no_reuse: bool,
        evenly_shuffled: bool,
        skip_confirm: bool
):
    """CLI for generating distance comparison True/False questions"""
    scene_stat_files = enum_files(scene_stat_file, '.json', skip_confirm)

    # validate export file, overwrite it if exists and confirmed
    print(f'Exporting questions to: {export_json_file}')
    if not confirm_overwrite_file(export_json_file):
        raise click.Abort('Aborted overwriting existing file')

    print(f'Expected to generate {n_questions * len(scene_stat_files)} questions total')

    # exclude the specified labels from the question generation
    exclude_labels = [label.strip() for label in exclude_labels.split(',')]
    print(f'Excluding labels: {exclude_labels}')

    print(f'{" Started generating distance comparison questions ":=^80}')
    for scene_file in tqdm(scene_stat_files, desc='Processing', unit='scene'):
        generator = DistanceCompareTFQGenerator(
            scene_stat_json_file=scene_file,
            export_json_file=export_json_file,
            exclude_labels=exclude_labels
        )
        generator.generate(
            n_questions=n_questions,
            is_strict_no_reuse=strict_no_reuse,
            is_evenly_shuffled_boolean=evenly_shuffled
        )

    print(f'{" Finished generating distance comparison questions ":=^80}')


if __name__ == '__main__':
    cli()
