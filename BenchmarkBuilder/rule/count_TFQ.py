import random
from typing import Optional, List

import click

from BenchmarkBuilder.rule.base import (
    PROMPT_TFQ_HINT_TEMPLATES,
    RuleBasedQuestionGenerator, TFQGeneratorMixin
)

# templates for count comparison TFQ
COUNT_COMPARE_TFQ_RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the count of <OBJ1> greater than the count of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is greater than the count of <OBJ2>? ',
            'Are there more <OBJ1> than <OBJ2>? ',
        ],
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the count of <OBJ1> less than the count of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is less than the count of <OBJ2>? ',
            'Are there fewer <OBJ1> than <OBJ2>? ',
        ],
    },
    '=': {
        'func': lambda x, y: x == y,
        'text': 'equal to',
        'templates': [
            'Is the count of <OBJ1> equal to the count of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is equal to the count of <OBJ2>? ',
            'Are there the same number of <OBJ1> as <OBJ2>? ',
        ],
    },
}

COUNT_COMPARE_TFQ_CoT_TEMPLATE = """Given the count of <OBJ1> as <OBJ1_COUNT> and the count of <OBJ2> as <OBJ2_COUNT>,
the count of <OBJ1> is <RELATION> the count of <OBJ2>.
Therefore, the answer is <<answer:<ANSWER>>>."""


class CountTFQGenerator(
    RuleBasedQuestionGenerator, TFQGeneratorMixin
):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str = './output/NUM-count-TFQ.json',
            excluded_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            excluded_labels=excluded_labels,
            question_type='RULE-count-TFQ',
        )

    def _form_question_dict(
            self,
            relation: str,
            preset_boolean: bool,
            label1: str, label2: str
    ) -> dict[str, dict[str, str | float | bool] | str | list[str]]:
        """Form multiple choice question for count comparison between two objects in the scene"""
        inst1s = self.scene_data.get_instances_by_label(label1)
        inst2s = self.scene_data.get_instances_by_label(label2)
        label1_count = len(inst1s)
        label2_count = len(inst2s)

        return {
            'meta': {
                'label1': {
                    'label': label1,
                    'ids': [inst.object_id for inst in inst1s],
                    'count': label1_count,
                },
                'label2': {
                    'label': label2,
                    'ids': [inst.object_id for inst in inst2s],
                    'count': label2_count,
                },
                'relation': relation,
                'preset_boolean': preset_boolean,
            },
            'prompt': random.choice(COUNT_COMPARE_TFQ_RELATION_DICT[relation]['templates']).replace(
                '<OBJ1>', label1).replace(
                '<OBJ2>', label2) + random.choice(PROMPT_TFQ_HINT_TEMPLATES).strip(),
            'caption': 'yes' if preset_boolean else 'no',
            'CoT_caption': COUNT_COMPARE_TFQ_CoT_TEMPLATE.replace(
                '<OBJ1>', label1).replace(
                '<OBJ2>', label2).replace(
                '<OBJ1_COUNT>', str(label1_count)).replace(
                '<OBJ2_COUNT>', str(label2_count)).replace(
                '<RELATION>', COUNT_COMPARE_TFQ_RELATION_DICT[relation]['text']).replace(
                '<ANSWER>', 'yes' if preset_boolean else 'no').replace(
                '\n', ' '),
            'ref_captions': [
                'Yes', 'Y', 'True', 'T', 'Correct',
            ] if preset_boolean else [
                'No', 'N', 'False', 'F', 'Incorrect',
            ],
        }

    def generate(
            self,
            n_questions: int,
            max_retries: int = 3,
            enforce_balanced_boolean: bool = True,
            allow_duplicate_pairs: bool = True,
    ) -> None:
        """Generate multiple choice questions for comparing the count of objects in the scene"""
        available_labels = self._available_inst_labels
        if len(available_labels) < 2:
            click.echo(f'[ERROR] Not enough instances found for generating {self.question_type}, '
                       f'possibly due to strict filter criteria for scene: {self.scene_data.scene_id}, '
                       f'aborting question generation...')
            return

        preset_booleans = self._generate_preset_booleans(n_questions, enforce_balanced_boolean)

        question_dicts = []
        remaining_labels = available_labels.copy()
        for idx in range(n_questions):
            if not remaining_labels:
                if allow_duplicate_pairs:
                    click.echo(f'[WARN] Not enough unique instances found for generating {self.question_type}, '
                               f'allowing duplicate pairs...')
                    remaining_labels = available_labels.copy()
                else:
                    click.echo(f'[ERROR] Not enough unique instances found for generating {self.question_type}, '
                               f'aborting question generation...')
                    break

            for attempt in range(1, max_retries + 1):
                relation = random.choice(list(COUNT_COMPARE_TFQ_RELATION_DICT.keys()))

                label1 = random.choice(remaining_labels)
                valid_label2s = [
                    label for label in remaining_labels
                    if label != label1 and
                       COUNT_COMPARE_TFQ_RELATION_DICT[relation]['func'](
                           len(self.scene_data.get_instances_by_label(label1)),
                           len(self.scene_data.get_instances_by_label(label))
                       ) == preset_booleans[idx]
                ]
                # retry or skip if no valid pair found
                if not valid_label2s:
                    if attempt == max_retries:
                        click.echo(f'[ERROR] Unable to find a valid pair for question {idx + 1} '
                                   f'after {max_retries} attempts, skipping...')
                        break
                    click.echo(f'[WARN] ({attempt}/{max_retries}) No valid pair found '
                               f'for question {idx + 1}, retrying...')
                    continue
                # select a valid label
                label2 = random.choice(valid_label2s)

                q_dict = self._form_question_dict(
                    relation, preset_booleans[idx], label1, label2
                )
                if q_dict not in question_dicts:
                    question_dicts.append(q_dict)
                    if not allow_duplicate_pairs:
                        remaining_labels.remove(label1)
                        remaining_labels.remove(label2)
                    break
                else:
                    click.echo(f'[WARN] Duplicate question found for question {idx + 1}, retrying...')

            click.echo(f'[ERROR] Unable to find a valid pair for {self.question_type} (Q{idx + 1}) '
                       f'after {max_retries} attempts, skipping...')

        self._export_question_dicts(question_dicts)
