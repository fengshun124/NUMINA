import random
from typing import Dict, Any

from BenchmarkBuilder.rule.base.base import (
    SingleObjectCandidateMixin, SAQMixin
)
from BenchmarkBuilder.rule.base.template import (
    PROMPT_SAQ_HINT_TEMPLATES, PROMPT_SAQ_CoT_HINT_TEMPLATE,
)
from BenchmarkBuilder.utils.scene import SceneInstance

VOLUME_SAQ_TEMPLATES = [
    'Can you calculate the volume of the bounding box of <OBJ> in cubic meters? ',
    'Can you estimate the volume of the bounding box of <OBJ> in cubic meters? ',
    'Please calculate the volume of the bounding box of <OBJ> in cubic meters. ',
    'Please estimate the volume of the bounding box of <OBJ> in cubic meters. ',
    'What is the volume of the bounding box of <OBJ> in cubic meters? '
]

VOLUME_SAQ_CoT_CAPTION_TEMPLATE = """Given the bounding box dimensions of the object along the X, Y, and Z axes
as <BBOX_X_LEN> m, <BBOX_Y_LEN> m, and <BBOX_Z_LEN> m respectively,
the volume of the bounding box is calculated as (length x width x height) yielding approximately
<<ANSWER>> cubic meters."""


class VolumeSAQGenerator(SAQMixin, SingleObjectCandidateMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_type = 'RULE-volume-SAQ'
        self.allow_repeated_objects = False

    @staticmethod
    def _custom_instance_filter(instance: SceneInstance) -> bool:
        # filter those with size less than 0.01 cubic meters
        if instance.bbox_volume < 0.02:
            return False
        # filter those with one dimension significantly smaller than the other two
        if min(instance.bbox_xyz_len) / max(instance.bbox_xyz_len) > .2:
            return False
        return True

    def _form_question_dict(self, **kwargs) -> Dict[str, Any]:
        instance = kwargs['candidate']
        label = instance.label

        # prepare the main proposition text
        base_prompt_text = (
            random.choice(VOLUME_SAQ_TEMPLATES)
            .replace('<OBJ>', label)
        )

        return {
            'meta': {
                'label': label,
                'id': [inst.object_id
                       for inst in self.scene_data.get_instances_by_label(label)],
                'bbox_xyz_min': instance.bbox_xyz_min,
                'bbox_xyz_max': instance.bbox_xyz_max,
                'bbox_xyz_len': instance.bbox_xyz_len,
                'bbox_volume': instance.bbox_volume,
            },
            'prompt': base_prompt_text + random.choice(PROMPT_SAQ_HINT_TEMPLATES),
            'CoT_prompt': base_prompt_text + PROMPT_SAQ_CoT_HINT_TEMPLATE,
            'caption': f'{round(instance.bbox_volume, 3):.2f}',
            'CoT_caption': (
                VOLUME_SAQ_CoT_CAPTION_TEMPLATE
                .replace('<BBOX_X_LEN>', f'{round(instance.bbox_xyz_len[0], 3):.2f}')
                .replace('<BBOX_Y_LEN>', f'{round(instance.bbox_xyz_len[1], 3):.2f}')
                .replace('<BBOX_Z_LEN>', f'{round(instance.bbox_xyz_len[2], 3):.2f}')
                .replace('<ANSWER>', f'{round(instance.bbox_volume, 3):.2f}')
            ),
            'ref_captions': [f'{round(instance.bbox_volume, 3):.2f}'],
        }
