import random

from BenchmarkBuilder.rule.base import (
    PROMPT_SAQ_HINT_TEMPLATES,
    DualLabelsBasedQuestionGenerator,
)

DISTANCE_SAQ_TEMPLATES = [
    'Can you estimate the distance between the <OBJ1> and <OBJ2> in the room in meters? ',
    'Can you calculate the distance between the <OBJ1> and <OBJ2> in the room in meters? ',
    'Please estimate the distance between the <OBJ1> and <OBJ2> in the room in meters. ',
    'Please calculate the distance between the <OBJ1> and <OBJ2> in the room in meters. ',
]


class DistanceSAQGenerator(DualLabelsBasedQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str = './output/NUM-distance-SAQ.json',
            excluded_labels: list[str] | None = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            excluded_labels=excluded_labels,
            allow_repeated_label1=False,
            allow_repeated_label2=False,
            question_type='RULE-distance-SAQ',
        )

    def _form_question_dict(
            self, label1: str, label2: str
    ) -> dict[str, dict[str, str | float | bool] | str | list[str]]:
        """Form short answer question for distance between two objects in the scene"""
        inst1 = self.scene_data.get_instances_by_label(label1)[0]
        inst2 = self.scene_data.get_instances_by_label(label2)[0]

        pairwise_distance = self.scene_data.get_pairwise_distance(
            inst1.object_id, inst2.object_id)

        return {
            'meta': {
                'obj1_id': inst1.object_id,
                'obj1_label': inst1.label,
                'obj2_id': inst2.object_id,
                'obj2_label': inst2.label,
                'distance': pairwise_distance,
            },
            'prompt': random.choice(DISTANCE_SAQ_TEMPLATES).replace(
                '<OBJ1>', inst1.label).replace(
                '<OBJ2>', inst2.label) + random.choice(PROMPT_SAQ_HINT_TEMPLATES),
            'caption': f'{pairwise_distance:>.2f}',
            'CoT_caption': f'<<answer:{pairwise_distance:>.2f}>>',
            'ref_captions': [
                f'{pairwise_distance:>.2f}',
                # f'{pairwise_distance:>.2f} meters',
                # f'{pairwise_distance:>.2f} m',
            ],
        }
