import random
from typing import Dict, Any

from .base.base import (
    DualObjectsCandidateMixin, FactValidationMixin
)
from .base.template import (
    PROMPT_FV_HINT_TEMPLATES,
    PROMPT_FV_CoT_HINT_TEMPLATE,
    CAPTION_FV_AFFIRMATIVES,
    CAPTION_FV_NEGATIVES,
)
from ..utils.scene import SceneInstance

VOLUME_COMPARE_FV_RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> greater than the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is greater than the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> greater than the one of <OBJ2>? ',
        ],
        'contrapositive': '<=',
    },
    '>=': {
        'func': lambda x, y: x >= y,
        'text': 'greater than or equal to',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> greater than or equal to the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is greater than or equal to the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> greater than or equal to the one of <OBJ2>? ',
        ],
        'contrapositive': '<',
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> less than the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is less than the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> less than the one of <OBJ2>? ',
        ],
        'contrapositive': '>=',
    },
    '<=': {
        'func': lambda x, y: x <= y,
        'text': 'less than or equal to',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> less than or equal to the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is less than or equal to the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> less than or equal to the one of <OBJ2>? ',
        ],
        'contrapositive': '>',
    },
    '=': {
        # relative error <= 5%
        'func': lambda x, y: abs(x - y) <= 0.05 * max(x, y),
        'text': 'approximately equal to',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> approximately equal to the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is approximately equal to the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> approximately equal to the one of <OBJ2>? ',
        ],
        'contrapositive': '!=',
    },
    '!=': {
        'func': lambda x, y: abs(x - y) > 0.05 * max(x, y),
        'text': 'not approximately equal to',
        'templates': [
            'Is the volume of the bounding box of <OBJ1> not approximately equal to the volume of the bounding box of <OBJ2>? ',
            'Can you tell if the volume of the bounding box of <OBJ1> is not approximately equal to the volume of the bounding box of <OBJ2>? ',
            'Is the size of the bounding box of <OBJ1> not approximately equal to the one of <OBJ2>? ',
        ],
        'contrapositive': '=',
    },
}

VOLUME_COMPARE_FV_CoT_CAPTION_TEMPLATE = """Given the volume of the bounding box of <OBJ1> as <OBJ1_VOLUME> cubic meters
and the volume of the bounding box of <OBJ2> as <OBJ2_VOLUME> cubic meters,
the volume of the bounding box of <OBJ1> is <BOOLEAN> <RELATION> the volume of the bounding box of <OBJ2>.
Therefore, the answer is <<ANSWER>>."""


class VolumeCompareFVGenerator(FactValidationMixin, DualObjectsCandidateMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_type = 'RULE-volume_compare-FV'
        self.allow_repeated_inst1s = False
        self.allow_repeated_inst2s = False

    @staticmethod
    def _custom_instance_filter(instance: SceneInstance) -> bool:
        # filter those with size less than 0.02 cubic meters
        if instance.bbox_volume < 0.02:
            return False
        # filter those with one dimension significantly smaller than the other two
        if min(instance.bbox_xyz_len) / max(instance.bbox_xyz_len) < 0.2:
            return False
        return True

    def _form_question_dict(self, **kwargs) -> Dict[str, Any]:
        inst1, inst2 = kwargs['candidate']
        preset_boolean = kwargs['preset_boolean']

        inst1_label, inst1_bbox_volume = inst1.label, inst1.bbox_volume
        inst2_label, inst2_bbox_volume = inst2.label, inst2.bbox_volume

        # find all relations that yield the intended boolean
        valid_relations = [
            rel for rel, info in VOLUME_COMPARE_FV_RELATION_DICT.items()
            if info['func'](inst1_bbox_volume, inst2_bbox_volume) == preset_boolean
        ]
        if valid_relations:
            relation = random.choice(valid_relations)
        else:
            # fallback to a random relation and then swap with its contrapositive
            relation = random.choice(
                list(VOLUME_COMPARE_FV_RELATION_DICT.keys()))
            if VOLUME_COMPARE_FV_RELATION_DICT[relation]['func'](
                    inst1_bbox_volume, inst2_bbox_volume) != preset_boolean:
                relation = VOLUME_COMPARE_FV_RELATION_DICT[relation]['contrapositive']

        # determine the contrapositive relation for the proposition
        contrapositive_relation = VOLUME_COMPARE_FV_RELATION_DICT[relation]['contrapositive']

        # prepare the main proposition text
        base_prompt_text = (
            random.choice(
                VOLUME_COMPARE_FV_RELATION_DICT[relation]['templates'])
            .replace('<OBJ1>', inst1_label)
            .replace('<OBJ2>', inst2_label)
        )
        prompt_caption = 'yes' if preset_boolean else 'no'
        base_prompt_suffix_text = random.choice(PROMPT_FV_HINT_TEMPLATES)

        # prepare the contrapositive proposition text
        cp_base_prompt_text = (
            random.choice(
                VOLUME_COMPARE_FV_RELATION_DICT[contrapositive_relation]['templates'])
            .replace('<OBJ1>', inst1_label)
            .replace('<OBJ2>', inst2_label)
        )
        cp_prompt_caption = 'yes' if not preset_boolean else 'no'

        # build the chain-of-thought texts.
        chain_of_thought_base_text = (
            VOLUME_COMPARE_FV_CoT_CAPTION_TEMPLATE
            .replace('<OBJ1>', inst1_label)
            .replace('<OBJ1_VOLUME>', f'{round(inst1_bbox_volume, 3):.2f}')
            .replace('<OBJ2>', inst2_label)
            .replace('<OBJ2_VOLUME>', f'{round(inst2_bbox_volume, 3):.2f}')
        )
        chain_of_thought_prompt_suffix_text = PROMPT_FV_CoT_HINT_TEMPLATE

        return {
            'meta': {
                'label1': {
                    'label': inst1_label,
                    'id': [
                        inst.object_id
                        for inst in self.scene_data.get_instances_by_label(inst1_label)
                    ],
                    'bbox_xyz_len': inst1.bbox_xyz_len,
                    'bbox_volume': inst1_bbox_volume,
                },
                'label2': {
                    'label': inst2_label,
                    'id': [
                        inst.object_id
                        for inst in self.scene_data.get_instances_by_label(inst2_label)
                    ],
                    'bbox_xyz_len': inst2.bbox_xyz_len,
                    'bbox_volume': inst2_bbox_volume,
                },
                'relation': relation,
                'cp_relation': contrapositive_relation,
                'preset_boolean': preset_boolean,
            },

            'prompt': base_prompt_text + base_prompt_suffix_text,
            'CoT_prompt': base_prompt_text + chain_of_thought_prompt_suffix_text,
            'caption': prompt_caption,
            'CoT_caption': (
                chain_of_thought_base_text
                .replace('<RELATION>', VOLUME_COMPARE_FV_RELATION_DICT[relation]['text'])
                .replace('<BOOLEAN>', '' if preset_boolean else 'not')
                .replace('<ANSWER>', prompt_caption)
            ),
            'ref_captions': CAPTION_FV_AFFIRMATIVES if preset_boolean else CAPTION_FV_NEGATIVES,

            'cp_prompt': cp_base_prompt_text + base_prompt_suffix_text,
            'cp_CoT_prompt': cp_base_prompt_text + chain_of_thought_prompt_suffix_text,
            'cp_caption': cp_prompt_caption,
            'cp_CoT_caption': (
                chain_of_thought_base_text
                .replace('<RELATION>', VOLUME_COMPARE_FV_RELATION_DICT[contrapositive_relation]['text'])
                .replace('<BOOLEAN>', '' if not preset_boolean else 'not')
                .replace('<ANSWER>', cp_prompt_caption)
            ),
            'cp_ref_captions': CAPTION_FV_AFFIRMATIVES if not preset_boolean else CAPTION_FV_NEGATIVES,
        }
