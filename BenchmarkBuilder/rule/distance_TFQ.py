import random
from itertools import combinations

import click

from BenchmarkBuilder.rule.base import (
    PROMPT_TFQ_HINT_TEMPLATES,
    RuleBasedQuestionGenerator,
    TFQGeneratorMixin,
)

DISTANCE_COMPARE_TFQ_RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the distance between <P1-OBJ1> and <P1-OBJ2> greater than the distance between <P2-OBJ1> and <P2-OBJ2>? ',
            'Is <P1-OBJ1> and <P1-OBJ2> further than <P2-OBJ1> and <P2-OBJ2>? '
        ]
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the distance between <P1-OBJ1> and <P1-OBJ2> less than the distance between <P2-OBJ1> and <P2-OBJ2>? ',
            'Is <P1-OBJ1> and <P1-OBJ2> closer than <P2-OBJ1> and <P2-OBJ2>? '
        ]
    },
    '=': {
        'func': lambda x, y: abs(x - y) < 0.1,
        'text': 'approximately equal to',
        'templates': [
            'Is the distance between <P1-OBJ1> and <P1-OBJ2> approximately the same as the distance between <P2-OBJ1> and <P2-OBJ2>? ',
            'Is <P1-OBJ1> and <P1-OBJ2> approximately as far as <P2-OBJ1> and <P2-OBJ2>? '
        ]
    }
}

DISTANCE_COMPARE_TFQ_CoT_TEMPLATE = """The distance between <P1-OBJ1> and <P1-OBJ2> is approximately <DIST1> meters.
The distance between <P2-OBJ1> and <P2-OBJ2> is approximately <DIST2> meters.
Since the distance between <P1-OBJ1> and <P1-OBJ2> is <RELATION> the distance between <P2-OBJ1> and <P2-OBJ2>,
the correct answer is <<answer:<ANSWER>>>.
"""


class DistanceTFQGenerator(
    RuleBasedQuestionGenerator, TFQGeneratorMixin
):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str = './output/NUM-distance_pair_compare-TFQ.json',
            excluded_labels: list[str] | None = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            excluded_labels=excluded_labels,
            question_type='RULE-distance_pair_compare-TFQ',
            allow_repeated_objs=False
        )

    def generate(
            self,
            n_questions: int,
            max_retries: int = 3,
            enforce_balanced_boolean: bool = True,
            allow_duplicate_pairs: bool = False,
    ) -> None:
        """Generate True/False questions for comparing the distance between pairs of objects in the scene"""
        available_labels = self._available_inst_labels
        if len(available_labels) < 3:
            click.echo(f'[ERROR] Not enough instances found for generating {self.question_type}, '
                       f'possibly due to strict filter criteria for scene: {self.scene_data.scene_id}, '
                       f'aborting question generation...')
            return
        available_pairs = list(combinations(available_labels, 2))

        preset_booleans = self._generate_preset_booleans(n_questions, enforce_balanced_boolean)

        question_dicts = []
        remaining_pairs = available_pairs.copy()
        for idx in range(n_questions):
            if not remaining_pairs:
                if allow_duplicate_pairs:
                    click.echo(f'[WARN] Not enough unique instances found for generating {self.question_type}, '
                               f'allowing duplicate pairs...')
                    remaining_pairs = available_pairs.copy()
                else:
                    click.echo(f'[ERROR] Not enough unique instances found for generating {self.question_type}, '
                               f'aborting question generation...')
                    break

            for attempt in range(1, max_retries + 1):
                relation = random.choice(list(DISTANCE_COMPARE_TFQ_RELATION_DICT.keys()))
                relation_func = DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['func']
                # select a pair of objects
                pair1_label1, pair1_label2 = random.choice(remaining_pairs)
                pairwise_distance1 = self.scene_data.get_pairwise_distance(
                    self.scene_data.get_instances_by_label(pair1_label1)[0].object_id,
                    self.scene_data.get_instances_by_label(pair1_label2)[0].object_id,
                )

                # select a pair of objects satisfying the relation
                valid_pair2s = [
                    (label1, label2) for label1, label2 in remaining_pairs
                    if (label1, label2) != (pair1_label1, pair1_label2)
                       and relation_func(
                        pairwise_distance1,
                        self.scene_data.get_pairwise_distance(
                            self.scene_data.get_instances_by_label(label1)[0].object_id,
                            self.scene_data.get_instances_by_label(label2)[0].object_id,
                        )
                    ) is preset_booleans[idx]
                ]

                # retry or skip if no valid pairs found
                if not valid_pair2s:
                    if attempt == max_retries:
                        click.echo(f'[ERROR] Unable to find a valid pair for question {idx + 1} '
                                   f'after {max_retries} attempts, skipping...')
                        break
                    click.echo(f'[WARN] ({attempt}/{max_retries}) No valid pair found '
                               f'for question {idx + 1}, retrying...')
                    continue
                # select a valid pair
                pair2_label1, pair2_label2 = random.choice(valid_pair2s)

                q_dict = self._form_question_dict(
                    relation, preset_booleans[idx],
                    pair1_label1, pair1_label2,
                    pair2_label1, pair2_label2,
                )
                if q_dict not in question_dicts:
                    question_dicts.append(q_dict)
                    if not allow_duplicate_pairs:
                        remaining_pairs.remove((pair1_label1, pair1_label2))
                        remaining_pairs.remove((pair2_label1, pair2_label2))
                    break
                else:
                    click.echo(f'[WARN] Duplicate question found for question {idx + 1}, retrying...')

            # skip if no valid pair found after max_retries
            click.echo(f'[ERROR] Unable to form valid pairs for {self.question_type} (Q{idx + 1})'
                       f'after {max_retries} attempts, skipping...')

        self._export_question_dicts(question_dicts)

    def _form_question_dict(
            self,
            relation: str,
            preset_boolean: bool,
            pair1_label1: str, pair1_label2: str,
            pair2_label1: str, pair2_label2: str,
    ) -> dict[str, dict[str, str | float | bool] | str | list[str]]:
        """Form True/False question for comparing the distance between pairs of objects in the scene"""
        inst1 = self.scene_data.get_instances_by_label(pair1_label1)[0]
        inst2 = self.scene_data.get_instances_by_label(pair1_label2)[0]
        inst3 = self.scene_data.get_instances_by_label(pair2_label1)[0]
        inst4 = self.scene_data.get_instances_by_label(pair2_label2)[0]

        pairwise_distance1 = self.scene_data.get_pairwise_distance(
            inst1.object_id, inst2.object_id)
        pairwise_distance2 = self.scene_data.get_pairwise_distance(
            inst3.object_id, inst4.object_id)

        return {
            'meta': {
                'pair1_obj1_id': inst1.object_id,
                'pair1_obj1_label': inst1.label,
                'pair1_obj2_id': inst2.object_id,
                'pair1_obj2_label': inst2.label,
                'pair2_obj1_id': inst3.object_id,
                'pair2_obj1_label': inst3.label,
                'pair2_obj2_id': inst4.object_id,
                'pair2_obj2_label': inst4.label,
                'distance1': pairwise_distance1,
                'distance2': pairwise_distance2,
                'relation': relation,
                'answer': DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['func'](
                    pairwise_distance1, pairwise_distance2),
            },
            'prompt': random.choice(DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['templates']).replace(
                '<P1-OBJ1>', inst1.label).replace(
                '<P1-OBJ2>', inst2.label).replace(
                '<P2-OBJ1>', inst3.label).replace(
                '<P2-OBJ2>', inst4.label) + random.choice(PROMPT_TFQ_HINT_TEMPLATES),
            'caption': 'yes' if preset_boolean else 'no',
            'CoT_caption': DISTANCE_COMPARE_TFQ_CoT_TEMPLATE.replace(
                '<P1-OBJ1>', inst1.label).replace(
                '<P1-OBJ2>', inst2.label).replace(
                '<P2-OBJ1>', inst3.label).replace(
                '<P2-OBJ2>', inst4.label).replace(
                '<DIST1>', f'{pairwise_distance1:.2f}').replace(
                '<DIST2>', f'{pairwise_distance2:.2f}').replace(
                '<RELATION>', DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['text']).replace(
                '<ANSWER>', 'yes' if preset_boolean else 'no').replace(
                '\n', ' '),
            'ref_captions': [
                'Yes', 'Y', 'True', 'T', 'Correct',
            ] if preset_boolean else [
                'No', 'N', 'False', 'F', 'Incorrect',
            ],
        }
