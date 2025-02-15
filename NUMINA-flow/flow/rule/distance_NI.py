import random
from typing import Dict, Any

from .base.base import (
    NumericalInferenceMixin, DualObjectsCandidateMixin
)
from .base.template import (
    PROMPT_NI_HINT_TEMPLATES, )

DISTANCE_NI_TEMPLATES = [
    'Can you estimate the distance between the <OBJ1> and <OBJ2> in the room in meters? ',
    'Can you calculate the distance between the <OBJ1> and <OBJ2> in the room in meters? ',
    'Please estimate the distance between the <OBJ1> and <OBJ2> in the room in meters. ',
    'Please calculate the distance between the <OBJ1> and <OBJ2> in the room in meters. ',
]


class DistanceNIGenerator(NumericalInferenceMixin, DualObjectsCandidateMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_type = 'RULE-distance-NI'
        self.allow_repeated_inst1s = False
        self.allow_repeated_inst2s = False

    def _form_question_dict(self, **kwargs) -> Dict[str, Any]:
        inst1, inst2 = kwargs['candidate']
        dist = self.scene_data.get_pairwise_distance(
            inst1.object_id, inst2.object_id)

        # prepare the main proposition text
        base_prompt_text = (
            random.choice(DISTANCE_NI_TEMPLATES)
            .replace('<OBJ1>', inst1.label)
            .replace('<OBJ2>', inst2.label)
        )

        return {
            'meta': {
                **{
                    inst_label: {
                        'label': inst.label,
                        'id': [i.object_id for i in self.scene_data.get_instances_by_label(inst.label)]
                    }
                    for inst_label, inst in {'inst1': inst1, 'inst2': inst2}.items()
                },
                'pairwise_distance': dist,
            },
            'prompt': base_prompt_text + random.choice(PROMPT_NI_HINT_TEMPLATES),
            'caption': f'{round(dist, 3):.2f}',
            'ref_captions': [f'{round(dist, 3):.2f}'],
        }
