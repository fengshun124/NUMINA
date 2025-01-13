import os
import random
import sys
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from BenchmarkBuilder.utils.scene import SceneInstance
from BenchmarkBuilder.rule.base import ObjectAttributeQuestionGenerator

PROMPT_QUESTION_TEMPLATES = [
    'Can you calculate the bounding box volume of the <OBJ> in the room in cubic meters? ',
    'Can you estimate the bounding box volume of the <OBJ> in the room in cubic meters? ',
    'Please calculate the volume of the bounding box of <OBJ> in cubic meters. ',
]

CoT_TEMPLATE = """Given the bounding box dimensions of the object along the X, Y, and Z axes 
as <BBOX_X_LEN> m, <BBOX_Y_LEN> m, and <BBOX_Z_LEN> m respectively, 
the volume of the bounding box is calculated as (length x width x height) yielding approximately  
<<answer:<BBOX_VOLUME>>> cubic meters."""


class VolumeSAQGenerator(ObjectAttributeQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str = './output/NUM-volume-SAQ.json',
            excluded_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            excluded_labels,
            require_singleton_objs=True
        )

    def _custom_instance_filter(self, instance: SceneInstance) -> bool:
        """Exclude objects with almost flat bounding boxes"""
        return min(instance.bbox_xyz_len) / max(instance.bbox_xyz_len) > .1

    def _form_question_dict(self, label: str) -> dict:
        """Form question for counting objects in the scene"""
        instance = self.scene_data.get_instances_by_label(label)[0]
        bbox_volume = instance.bbox_volume

        return {
            'scene_id': self.scene_data.scene_id,
            'count_meta': {
                'label': label,
                'obj_id': instance.object_id,
                'bbox_xyz_min': instance.bbox_xyz_min,
                'bbox_xyz_max': instance.bbox_xyz_max,
                'bbox_xyz_len': instance.bbox_xyz_len,
                'bbox_volume': bbox_volume
            },
            'prompt': random.choice(PROMPT_QUESTION_TEMPLATES).replace('<OBJ>', label),
            'caption': f'{bbox_volume:>.2f}',
            'CoT_caption': CoT_TEMPLATE.replace('\n', ' ').replace(
                '<BBOX_X_LEN>', f'{instance.bbox_xyz_len[0]:>.2f}'
            ).replace(
                '<BBOX_Y_LEN>', f'{instance.bbox_xyz_len[1]:>.2f}'
            ).replace(
                '<BBOX_Z_LEN>', f'{instance.bbox_xyz_len[2]:>.2f}'
            ).replace(
                '<BBOX_VOLUME>', f'{bbox_volume:>.2f}'
            ),
            'question_type': 'Rule-ShortAnswer'
        }
