import random
from typing import Dict, Any

from BenchmarkBuilder.rule.base.base import (
    TFQMixin, DualObjectPairsCandidateMixin
)
from BenchmarkBuilder.rule.base.template import (
    PROMPT_TFQ_HINT_TEMPLATES,
    PROMPT_TFQ_CoT_HINT_TEMPLATE,
    CAPTION_TFQ_AFFIRMATIVES,
    CAPTION_TFQ_NEGATIVES,
)

DISTANCE_COMPARE_TFQ_RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the distance between <OBJ1A> and <OBJ1B> greater than the distance between <OBJ2A> and <OBJ2B>? ',
            'Can you tell if the distance between <OBJ1A> and <OBJ1B> is greater than the distance between <OBJ2A> and <OBJ2B>? ',
            'Is the distance between <OBJ1A> and <OBJ1B> greater than the one between <OBJ2A> and <OBJ2B>? ',
        ],
        'contrapositive': '<=',
    },
    '>=': {
        'func': lambda x, y: x >= y,
        'text': 'greater than or equal to',
        'templates': [
            'Is the distance between <OBJ1A> and <OBJ1B> greater than or equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Can you tell if the distance between <OBJ1A> and <OBJ1B> is greater than or equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Is the distance between <OBJ1A> and <OBJ1B> greater than or equal to the one between <OBJ2A> and <OBJ2B>? ',
        ],
        'contrapositive': '<',
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the distance between <OBJ1A> and <OBJ1B> less than the distance between <OBJ2A> and <OBJ2B>? ',
            'Can you tell if the distance between <OBJ1A> and <OBJ1B> is less than the distance between <OBJ2A> and <OBJ2B>? ',
            'Is the distance between <OBJ1A> and <OBJ1B> less than the one between <OBJ2A> and <OBJ2B>? ',
        ],
        'contrapositive': '>=',
        'contrapositive_text': 'greater than or equal to',
    },
    '<=': {
        'func': lambda x, y: x <= y,
        'text': 'less than or equal to',
        'templates': [
            'Is the distance between <OBJ1A> and <OBJ1B> less than or equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Can you tell if the distance between <OBJ1A> and <OBJ1B> is less than or equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Is the distance between <OBJ1A> and <OBJ1B> less than or equal to the one between <OBJ2A> and <OBJ2B>? ',
        ],
        'contrapositive': '>',
        'contrapositive_text': 'greater than',
    },
    '=': {
        # relative error <= 5%
        'func': lambda x, y: abs(x - y) <= 0.05 * max(x, y),
        'text': 'approximately equal to',
        'templates': [
            'Is the distance between <OBJ1A> and <OBJ1B> approximately equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Can you tell if the distance between <OBJ1A> and <OBJ1B> is approximately equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Is the distance between <OBJ1A> and <OBJ1B> approximately equal to the one between <OBJ2A> and <OBJ2B>? ',
        ],
        'contrapositive': '!=',
    },
    '!=': {
        'func': lambda x, y: abs(x - y) > 0.05 * max(x, y),
        'text': 'not approximately equal to',
        'templates': [
            'Is the distance between <OBJ1A> and <OBJ1B> not approximately equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Can you tell if the distance between <OBJ1A> and <OBJ1B> is not approximately equal to the distance between <OBJ2A> and <OBJ2B>? ',
            'Is the distance between <OBJ1A> and <OBJ1B> not approximately equal to the one between <OBJ2A> and <OBJ2B>? ',
        ],
        'contrapositive': '=',
    },
}

DISTANCE_COMPARE_TFQ_CoT_CAPTION_TEMPLATE = """The distance between <OBJ1A> and <OBJ1B> is approximately <DIST1> meters. 
The distance between <OBJ2A> and <OBJ2B> is approximately <DIST2> meters.
As the distance between <OBJ1A> and <OBJ1B> is <BOOLEAN> <RELATION> the distance between <OBJ2A> and <OBJ2B>,
the correct answer is <<ANSWER>>.
"""


class DistanceCompareTFQGenerator(TFQMixin, DualObjectPairsCandidateMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_type = 'RULE-distance_compare-TFQ'
        self.allow_repeated_inst1as = False
        self.allow_repeated_inst1bs = False
        self.allow_repeated_inst2as = False
        self.allow_repeated_inst2bs = False

    def _form_question_dict(self, **kwargs) -> Dict[str, Any]:
        (inst1a, inst1b), (inst2a, inst2b) = kwargs['candidate']
        preset_boolean = kwargs['preset_boolean']

        # get the distance between the object pairs
        dist1 = self.scene_data.get_pairwise_distance(inst1a.object_id, inst1b.object_id)
        dist2 = self.scene_data.get_pairwise_distance(inst2a.object_id, inst2b.object_id)

        # find all relations that yield the intended boolean
        valid_relations = [
            rel for rel, info in DISTANCE_COMPARE_TFQ_RELATION_DICT.items()
            if info['func'](dist1, dist2) == preset_boolean
        ]
        if valid_relations:
            relation = random.choice(valid_relations)
        else:
            # fallback to a random relation and then swap with its contrapositive
            relation = random.choice(list(DISTANCE_COMPARE_TFQ_RELATION_DICT.keys()))
            if DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['func'](
                    dist1, dist2) != preset_boolean:
                relation = DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['contrapositive']

        # determine the contrapositive relation for the proposition
        contrapositive_relation = DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['contrapositive']

        # prepare the main proposition text
        base_prompt_text = (
            random.choice(DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['templates'])
            .replace('<OBJ1A>', inst1a.label)
            .replace('<OBJ1B>', inst1b.label)
            .replace('<OBJ2A>', inst2a.label)
            .replace('<OBJ2B>', inst2b.label)
        )
        prompt_caption = 'yes' if preset_boolean else 'no'
        base_prompt_suffix_text = random.choice(PROMPT_TFQ_HINT_TEMPLATES)

        # prepare the contrapositive proposition text
        cp_base_prompt_text = (
            random.choice(DISTANCE_COMPARE_TFQ_RELATION_DICT[contrapositive_relation]['templates'])
            .replace('<OBJ1A>', inst1a.label)
            .replace('<OBJ1B>', inst1b.label)
            .replace('<OBJ2A>', inst2a.label)
            .replace('<OBJ2B>', inst2b.label)
        )
        cp_prompt_caption = 'yes' if not preset_boolean else 'no'

        # build the chain-of-thought texts
        chain_of_thought_base_text = (
            DISTANCE_COMPARE_TFQ_CoT_CAPTION_TEMPLATE
            .replace('<OBJ1A>', inst1a.label)
            .replace('<OBJ1B>', inst1b.label)
            .replace('<OBJ2A>', inst2a.label)
            .replace('<OBJ2B>', inst2b.label)
            .replace('<DIST1>', f'{round(dist1, 3):.2f}')
            .replace('<DIST2>', f'{round(dist2, 3):.2f}')
        )
        chain_of_thought_prompt_suffix_text = PROMPT_TFQ_CoT_HINT_TEMPLATE

        return {
            'meta': {
                'pair1': {
                    **{
                        inst_label: {
                            'label': inst.label,
                            'id': [i.object_id for i in self.scene_data.get_instances_by_label(inst.label)]
                        }
                        for inst_label, inst in {'inst1a': inst1a, 'inst1b': inst1b}.items()
                    },
                    'pairwise_distance': dist1,
                },
                'pair2': {
                    **{
                        inst_label: {
                            'label': inst.label,
                            'id': [i.object_id for i in self.scene_data.get_instances_by_label(inst.label)]
                        }
                        for inst_label, inst in {'inst2a': inst2a, 'inst2b': inst2b}.items()
                    },
                    'pairwise_distance': dist2,
                },
                'preset_boolean': preset_boolean,
                'relation': relation,
                'cp_relation': contrapositive_relation,
            },

            'prompt': base_prompt_text + base_prompt_suffix_text,
            'CoT_prompt': base_prompt_text + chain_of_thought_prompt_suffix_text,
            'caption': prompt_caption,
            'CoT_caption': (
                chain_of_thought_base_text
                .replace('<RELATION>', DISTANCE_COMPARE_TFQ_RELATION_DICT[relation]['text'])
                .replace('<BOOLEAN>', '' if preset_boolean else 'not')
                .replace('<ANSWER>', prompt_caption)
            ),
            'ref_captions': CAPTION_TFQ_AFFIRMATIVES if preset_boolean else CAPTION_TFQ_NEGATIVES,

            'cp_prompt': cp_base_prompt_text + base_prompt_suffix_text,
            'cp_CoT_prompt': cp_base_prompt_text + chain_of_thought_prompt_suffix_text,
            'cp_caption': cp_prompt_caption,
            'cp_CoT_caption': (
                chain_of_thought_base_text
                .replace('<RELATION>', DISTANCE_COMPARE_TFQ_RELATION_DICT[contrapositive_relation]['text'])
                .replace('<BOOLEAN>', '' if not preset_boolean else 'not')
                .replace('<ANSWER>', cp_prompt_caption)
            ),
            'cp_ref_captions': CAPTION_TFQ_AFFIRMATIVES if not preset_boolean else CAPTION_TFQ_NEGATIVES,
        }
