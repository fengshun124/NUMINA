import random
from typing import List, Dict, Any

from .base.base import (
    SingleObjectCandidateMixin, NumericalInferenceMixin
)
from .base.template import (
    PROMPT_NI_HINT_TEMPLATES,
)

COUNT_NI_TEMPLATES = [
    'Can you count the number of <OBJ> in the room? ',
    'Can you count the <OBJ> in the room? ',
    'Can you tell me how many <OBJ> there are in the room? ',
    'Can you tell me the number of <OBJ> in the room? ',
    'Count the number of <OBJ> in the room. ',
    'How many <OBJ> can you see in the room? ',
    'How many <OBJ> are there in the room? ',
    'How many <OBJ> are in the room? ',
    'Please count the number of <OBJ> in the room.'
]


class QuantityNIGenerator(NumericalInferenceMixin, SingleObjectCandidateMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_type = 'RULE-count-NI'
        self.allow_repeated_objects = True

    def _get_candidates(self) -> List[str]:
        return list(set(
            inst.label for inst in super()._get_candidates()
        ))

    def _form_question_dict(self, **kwargs) -> Dict[str, Any]:
        label = kwargs['candidate']
        instances = self.scene_data.get_instances_by_label(label)
        obj_count = len(instances)

        return {
            'meta': {
                'label': label,
                'obj_ids': [inst.object_id for inst in instances],
            },
            'prompt': (
                random.choice(COUNT_NI_TEMPLATES)
                .replace('<OBJ>', label)
                + random.choice(PROMPT_NI_HINT_TEMPLATES)
            ),
            'caption': str(obj_count),
            'ref_captions': [obj_count],
        }
