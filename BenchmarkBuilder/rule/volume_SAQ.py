import random
from typing import Optional, List

from BenchmarkBuilder.rule.base import (
    PROMPT_SAQ_HINT_TEMPLATES,
    SingleLabelBasedQuestionGenerator
)
from BenchmarkBuilder.utils.scene import SceneInstance

# template for volume SAQ
VOLUME_SAQ_TEMPLATES = [
    'Can you calculate the volume of the bounding box of <OBJ> in cubic meters? ',
    'Can you estimate the volume of the bounding box of <OBJ> in cubic meters? ',
    'Please calculate the volume of the bounding box of <OBJ> in cubic meters. ',
    'Please estimate the volume of the bounding box of <OBJ> in cubic meters. ',
    'What is the volume of the bounding box of <OBJ> in cubic meters? '
]

VOLUME_SAQ_CoT_TEMPLATE = """Given the bounding box dimensions of the object along the X, Y, and Z axes
as <BBOX_X_LEN> m, <BBOX_Y_LEN> m, and <BBOX_Z_LEN> m respectively,
the volume of the bounding box is calculated as (length x width x height) yielding approximately
<<answer:<BBOX_VOLUME>>> cubic meters."""


class VolumeSAQGenerator(SingleLabelBasedQuestionGenerator):
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
            allow_repeated_objs=False,
            question_type='RULE-volume-SAQ',
        )

    def _custom_instance_filter(self, instance: SceneInstance) -> bool:
        """Exclude objects with almost flat bounding boxes"""
        return min(instance.bbox_xyz_len) / max(instance.bbox_xyz_len) > .2

    def _form_question_dict(self, label: str) -> dict:
        """Form question for counting objects in the scene"""
        instance = self.scene_data.get_instances_by_label(label)[0]
        bbox_volume = instance.bbox_volume

        return {
            'meta': {
                'label': instance.label,
                'obj_id': instance.object_id,
                'bbox_xyz_min': instance.bbox_xyz_min,
                'bbox_xyz_max': instance.bbox_xyz_max,
                'bbox_xyz_len': instance.bbox_xyz_len,
                'bbox_volume': bbox_volume
            },
            'prompt': random.choice(VOLUME_SAQ_TEMPLATES).replace(
                '<OBJ>', instance.label) + random.choice(PROMPT_SAQ_HINT_TEMPLATES),
            'caption': f'{bbox_volume:.2f}',
            'CoT_caption': VOLUME_SAQ_CoT_TEMPLATE.replace('\n', ' ').replace(
                '<BBOX_X_LEN>', f'{instance.bbox_xyz_len[0]:.2f}'
            ).replace(
                '<BBOX_Y_LEN>', f'{instance.bbox_xyz_len[1]:.2f}'
            ).replace(
                '<BBOX_Z_LEN>', f'{instance.bbox_xyz_len[2]:.2f}'
            ).replace(
                '<BBOX_VOLUME>', f'{bbox_volume:.2f}'
            ),
            'ref_captions': [
                f'{bbox_volume:.2f}',
                f'{bbox_volume:.2f} cubic meters',
                f'{bbox_volume:.2f} m^3',
                f'{bbox_volume:.2f} m3',
            ],
        }
