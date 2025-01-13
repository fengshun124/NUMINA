import os
import random
import sys

import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from BenchmarkBuilder.rule.base import ObjectPairwiseDistanceQuestionGenerator

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

CoT_TEMPLATE = """The distance between <OBJ1> and <OBJ2> is approximately <DIST1> meters.
The distance between <OBJ3> and <OBJ4> is approximately <DIST2> meters.
Since the distance between <OBJ1> and <OBJ2> is <RELATION> the distance between <OBJ3> and <OBJ4>,
the correct answer is <<answer:<ANSWER>>>.
"""


class DistanceCompareTFQGenerator(ObjectPairwiseDistanceQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str = './output/NUM-dist_compare-TFQ.json',
            excluded_labels: list[str] = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            excluded_labels
        )

    def generate(
            self,
            n_questions: int,
            max_retries: int = 20,
            enforce_unique_pairs: bool = False,
            is_evenly_shuffled_boolean: bool = False,
    ) -> None:
        """Generate distance comparison questions"""
        self._validate_args(n_questions, max_retries)
        available_pairs = list(self._pairwise_distance_dict.keys())

        # shuffle boolean values for True/False answers
        preset_booleans = [i % 2 == 0 for i in range(n_questions)] if is_evenly_shuffled_boolean \
            else random.choices([True, False], k=n_questions)

        question_dicts = []
        remaining_pairs = available_pairs.copy()

        for _ in range(max_retries):
            if len(question_dicts) >= n_questions:
                break

            # stop if all pairs used in strict mode
            if enforce_unique_pairs and not remaining_pairs:
                break

            # select a distance relation
            relation = random.choice(list(RELATION_DICT.keys()))
            # select random pair
            pair1 = random.choice(remaining_pairs)
            dist1 = self._pairwise_distance_dict[pair1]
            # select another pair with different distance satisfying the relation
            preset_boolean = preset_booleans[len(question_dicts)]
            valid_pairs = [p for p in remaining_pairs if p != pair1 and
                           ((dist1 > self._pairwise_distance_dict[p]) == preset_boolean)]
            if not valid_pairs:
                continue
            pair2 = random.choice(valid_pairs)

            # remove used pairs if strict unique mode
            if enforce_unique_pairs:
                remaining_pairs.remove(pair1)
                remaining_pairs.remove(pair2)

            question_dict = self._form_question_dict(
                pair1=pair1, pair2=pair2,
                relation=relation,
                answer_boolean=preset_boolean
            )

            if question_dict not in question_dicts:
                question_dicts.append(question_dict)

        if len(question_dicts) < n_questions:
            click.echo(f'[WARN] Generated only {len(question_dicts)} / {n_questions} questions '
                       f'as all {len(available_pairs)} available pairs are used '
                       f'before reaching target question count.')
        self._export_question_dicts(question_dicts)

    def _validate_args(
            self,
            n_questions: int,
            max_retries: int
    ) -> None:
        if n_questions <= 0:
            raise ValueError('Number of questions should be positive')
        if max_retries <= 0:
            raise ValueError('Maximum retries should be positive')
        if n_questions % 2 != 0:
            click.echo('[WARN] Odd number of questions will NOT '
                       'leads to even distribution of True/False')

    def _form_question_dict(
            self,
            pair1: tuple[str, str],
            pair2: tuple[str, str],
            relation: str,
            answer_boolean: bool
    ) -> dict:
        obj1_id, obj2_id = pair1
        obj3_id, obj4_id = pair2

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
                'obj_pair1_key': f'{obj1_id}-{obj2_id}',
                'obj_pair1_distance': self.scene_data.get_pairwise_distance(obj1_id, obj2_id),
                'obj_pair2_key': f'{obj3_id}-{obj4_id}',
                'obj_pair2_distance': self.scene_data.get_pairwise_distance(obj3_id, obj4_id),
            },
            'prompt': random.choice(RELATION_DICT[relation]['templates']).replace(
                '<OBJ1>', obj1_label).replace(
                '<OBJ2>', obj2_label).replace(
                '<OBJ3>', obj3_label).replace(
                '<OBJ4>', obj4_label).replace(
                '<RELATION>', RELATION_DICT[relation]['text']
            ),
            'caption': str(answer_boolean),
            'CoT_caption': CoT_TEMPLATE.replace('\n', ' ').replace(
                '<OBJ1>', obj1_label).replace(
                '<OBJ2>', obj2_label).replace(
                '<OBJ3>', obj3_label).replace(
                '<OBJ4>', obj4_label).replace(
                '<DIST1>', f'{self.scene_data.get_pairwise_distance(obj1_id, obj2_id):>.1f}').replace(
                '<DIST2>', f'{self.scene_data.get_pairwise_distance(obj3_id, obj4_id):>.1f}').replace(
                '<RELATION>', RELATION_DICT[relation]['text']).replace(
                '<ANSWER>', str(answer_boolean)).replace(
                # line break for CoT readability
                '\n', ' '
            ),
            'question_type': 'Rule-TrueFalse'
        }
