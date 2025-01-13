import os
import random
import sys

import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from BenchmarkBuilder.rule.base import ObjectPairwiseDistanceQuestionGenerator

PROMPT_QUESTION_TEMPLATES = [
    'Can you estimate the distance between the <OBJ1> and <OBJ2> in the room in meters? ',
    'Can you calculate the distance between the <OBJ1> and <OBJ2> in the room in meters? ',
    'Please estimate the distance between the <OBJ1> and <OBJ2> in the room in meters. ',
    'Please calculate the distance between the <OBJ1> and <OBJ2> in the room in meters. ',
]


class DistanceSAQGenerator(ObjectPairwiseDistanceQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str = './output/NUM-distance-SAQ.json',
            excluded_labels: list[str] | None = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            excluded_labels,
        )

    def generate(
            self,
            n_questions: int,
            max_retries: int = 20,
            enforce_unique_pairs: bool = False,
            **kwargs,
    ) -> None:
        """Generate counting questions"""
        self._validate_args(n_questions, max_retries)
        available_obj_pair_dict = self._pairwise_distance_dict
        if not available_obj_pair_dict:
            click.echo(
                '[WARN] No available object pairs found for generating questions, NO questions will be generated.')
            return

        question_dicts = []
        used_pairs = set()
        remaining_pairs = set(available_obj_pair_dict.keys())

        retries = 0
        while len(question_dicts) < n_questions and retries < max_retries:
            if enforce_unique_pairs and not remaining_pairs:
                break

            pair = random.choice(list(remaining_pairs if enforce_unique_pairs else available_obj_pair_dict.keys()))
            if enforce_unique_pairs:
                remaining_pairs.remove(pair)

            question_dict = self._form_question_dict(pair)
            if question_dict not in question_dicts:
                question_dicts.append(question_dict)
            else:
                click.echo(f'[WARN] Duplicate question generated for pair: {pair}, skipping...')
            used_pairs.add(pair)
            retries += 1

        if len(question_dicts) < n_questions:
            click.echo(f'[WARN] Generated only {len(question_dicts)} / {n_questions} questions'
                       f'as all {len(available_obj_pair_dict.keys())} available pairs are used '
                       f'before reaching target question count.')
        self._export_question_dicts(question_dicts)

    def _form_question_dict(self, pair: tuple[str, str]) -> dict:
        """Form question for counting objects in the scene"""
        obj_inst1 = self.scene_data.get_instance_by_object_id(pair[0])
        obj_inst2 = self.scene_data.get_instance_by_object_id(pair[1])

        distance = self.scene_data.get_pairwise_distance(
            obj_id1=obj_inst1.object_id,
            obj_id2=obj_inst2.object_id
        )

        return {
            'scene_id': self.scene_data.scene_id,
            'count_meta': {
                'obj1_id': obj_inst1.object_id,
                'obj1_label': obj_inst1.label,
                'obj2_id': obj_inst2.object_id,
                'obj2_label': obj_inst2.label,
                'distance': self.scene_data.get_pairwise_distance(
                    obj_id1=obj_inst1.object_id,
                    obj_id2=obj_inst2.object_id
                ),
            },
            'prompt': random.choice(PROMPT_QUESTION_TEMPLATES).replace(
                '<OBJ1>', obj_inst1.label).replace(
                '<OBJ2>', obj_inst2.label),
            'caption': f'{distance:>.2f}',
            'CoT_caption': f'<<answer:{distance:>.2f}>>',
            'question_type': 'Rule-ShortAnswer'
        }
