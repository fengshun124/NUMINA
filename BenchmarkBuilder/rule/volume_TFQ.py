import random
from typing import Optional, List

import click

from BenchmarkBuilder.rule.base import (
    PROMPT_TFQ_HINT_TEMPLATES,
    RuleBasedQuestionGenerator, TFQGeneratorMixin
)
from BenchmarkBuilder.utils.scene import SceneInstance

# templates for volume comparison TFQ
VOLUME_COMPARE_TFQ_RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> greater than the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is greater than the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> greater than the one of <OBJ2>? ',
        ]
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> less than the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is less than the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> less than the one of <OBJ2>? ',
        ]
    },
    '=': {
        'func': lambda x, y: abs(x - y) < 0.02,
        'text': 'approximately equal to',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> approximately equal to the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is approximately equal to the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> approximately equal to the one of <OBJ2>? ',
            'Is the volume of the bounding box of <OBJ1> the same as the volume of the bounding box of <OBJ2>? ',
        ]
    }
}

VOLUME_COMPARE_TFQ_CoT_TEMPLATE = """Given the volume of the bounding box of <OBJ1> as <OBJ1_VOLUME> cubic meters
and the volume of the bounding box of <OBJ2> as <OBJ2_VOLUME> cubic meters,
the volume of the bounding box of <OBJ1> is <RELATION> the volume of the bounding box of <OBJ2>.
Therefore, the answer is <<answer:<ANSWER>>>."""


class VolumeTFQGenerator(
    RuleBasedQuestionGenerator, TFQGeneratorMixin
):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str = './output/NUM-volume-SAQ.json',
            excluded_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            excluded_labels=excluded_labels,
            question_type='RULE-volume-TFQ',
        )

    def _custom_instance_filter(self, instance: SceneInstance) -> bool:
        """Exclude objects with almost flat bounding boxes"""
        return min(instance.bbox_xyz_len) / max(instance.bbox_xyz_len) > .2

    def _form_question_dict(
            self,
            relation: str,
            preset_boolean: bool,
            label1: str, label2: str
    ) -> dict[str, dict[str, str | float | bool] | str | list[str]]:
        """Form True/False question for comparing the volume of objects in the scene"""
        inst1 = self.scene_data.get_instances_by_label(label1)[0]
        inst2 = self.scene_data.get_instances_by_label(label2)[0]
        volume1 = inst1.bbox_volume
        volume2 = inst2.bbox_volume

        return {
            'meta': {
                'obj1': {
                    'label': label1,
                    'obj_id': inst1.object_id,
                    'bbox_xyz_len': inst1.bbox_xyz_len,
                    'bbox_volume': volume1,
                },
                'obj2': {
                    'label': label2,
                    'obj_id': inst2.object_id,
                    'bbox_xyz_len': inst2.bbox_xyz_len,
                    'bbox_volume': volume2,
                },
                'relation': relation,
                'preset_boolean': preset_boolean,
            },
            'prompt': (random.choice(VOLUME_COMPARE_TFQ_RELATION_DICT[relation]['templates']).replace(
                '<OBJ1>', inst1.label).replace(
                '<OBJ2>', inst2.label) + random.choice(PROMPT_TFQ_HINT_TEMPLATES)).strip(),
            'caption': 'yes' if preset_boolean else 'no',
            'CoT_caption': VOLUME_COMPARE_TFQ_CoT_TEMPLATE.replace(
                '<OBJ1>', inst1.label).replace(
                '<OBJ1_VOLUME>', f'{volume1:.2f}').replace(
                '<OBJ2>', inst2.label).replace(
                '<OBJ2_VOLUME>', f'{volume2:.2f}').replace(
                '<RELATION>', VOLUME_COMPARE_TFQ_RELATION_DICT[relation]['text']).replace(
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
        """Generate True/False questions for comparing the volume of objects in the scene"""
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
                relation = random.choice(list(VOLUME_COMPARE_TFQ_RELATION_DICT.keys()))

                label1 = random.choice(remaining_labels)
                valid_label2s = [
                    label for label in remaining_labels
                    if label != label1 and
                       VOLUME_COMPARE_TFQ_RELATION_DICT[relation]['func'](
                           self.scene_data.get_instances_by_label(label1)[0].bbox_volume,
                           self.scene_data.get_instances_by_label(label)[0].bbox_volume
                       ) == preset_booleans[idx]
                ]
                # retry or skip if no valid pair found
                if not valid_label2s:
                    if attempt == max_retries:
                        click.echo(f'[ERROR] Unable to find a valid pair for {self.question_type} {idx + 1} '
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
