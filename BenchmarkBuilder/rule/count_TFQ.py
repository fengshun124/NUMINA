import random
from typing import Dict, Any, List, Tuple

from BenchmarkBuilder.rule.base.base import (
    DualObjectsCandidateMixin, TFQMixin
)
from BenchmarkBuilder.rule.base.template import (
    PROMPT_TFQ_HINT_TEMPLATES,
    PROMPT_TFQ_CoT_HINT_TEMPLATE,
    CAPTION_TFQ_AFFIRMATIVES,
    CAPTION_TFQ_NEGATIVES,
)

COUNT_COMPARE_TFQ_RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the number of <OBJ1> greater than the number of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is greater than the count of <OBJ2>? ',
            'Are there more <OBJ1> than <OBJ2>? ',
        ],
        'contrapositive': '<=',
    },
    '>=': {
        'func': lambda x, y: x >= y,
        'text': 'greater than or equal to',
        'templates': [
            'Is the number of <OBJ1> greater than or equal to the number of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is greater than or equal to the count of <OBJ2>? ',
            'Are there more <OBJ1> than or equal to <OBJ2>? ',
        ],
        'contrapositive': '<',
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the number of <OBJ1> less than the number of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is less than the count of <OBJ2>? ',
            'Are there fewer <OBJ1> than <OBJ2>? ',
        ],
        'contrapositive': '>=',
        'contrapositive_text': 'greater than or equal to',
    },
    '<=': {
        'func': lambda x, y: x <= y,
        'text': 'less than or equal to',
        'templates': [
            'Is the number of <OBJ1> less than or equal to the number of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is less than or equal to the count of <OBJ2>? ',
            'Are there fewer <OBJ1> than or equal to <OBJ2>? ',
        ],
        'contrapositive': '>',
        'contrapositive_text': 'greater than',
    },
    '=': {
        'func': lambda x, y: x == y,
        'text': 'equal to',
        'templates': [
            'Is the number of <OBJ1> equal to the number of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is equal to the count of <OBJ2>? ',
            'Are there the same number of <OBJ1> as <OBJ2>? ',
        ],
        'contrapositive': '!=',
    },
    '!=': {
        'func': lambda x, y: x != y,
        'text': 'not equal to',
        'templates': [
            'Is the number of <OBJ1> not equal to the number of <OBJ2>? ',
            'Can you tell if the count of <OBJ1> is not equal to the count of <OBJ2>? ',
            'Are there different numbers of <OBJ1> and <OBJ2>? ',
        ],
        'contrapositive': '=',
    },
}

COUNT_COMPARE_TFQ_CoT_CAPTION_TEMPLATE = """Given the count of <OBJ1> as <OBJ1_COUNT> and the count of <OBJ2> as <OBJ2_COUNT>,
the count of <OBJ1> is <BOOLEAN> <RELATION> the count of <OBJ2>.
Therefore, the answer is <<ANSWER>>"""


class CountCompareTFQGenerator(TFQMixin, DualObjectsCandidateMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_type = 'RULE-count_compare-TFQ'
        self.allow_repeated_inst1s = True
        self.allow_repeated_inst2s = True

    def _get_candidates(self) -> List[Tuple[str, str]]:
        candidates = super()._get_candidates()
        return [
            (inst1.label, inst2.label)
            for inst1, inst2 in candidates
            if inst1.label != inst2.label
        ]

    def _form_question_dict(self, **kwargs) -> Dict[str, Any]:
        label1, label2 = kwargs['candidate']
        preset_boolean = kwargs['preset_boolean']

        label1_instances = self.scene_data.get_instances_by_label(label1)
        label2_instances = self.scene_data.get_instances_by_label(label2)
        label1_inst_count = len(label1_instances)
        label2_inst_count = len(label2_instances)

        # find all relations that yield the intended boolean
        valid_relations = [
            rel for rel, info in COUNT_COMPARE_TFQ_RELATION_DICT.items()
            if info['func'](label1_inst_count, label2_inst_count) == preset_boolean
        ]
        if valid_relations:
            relation = random.choice(valid_relations)
        else:
            # fallback to a random relation and then swap with its contrapositive
            relation = random.choice(list(COUNT_COMPARE_TFQ_RELATION_DICT.keys()))
            if COUNT_COMPARE_TFQ_RELATION_DICT[relation]['func'](
                    label1_inst_count, label2_inst_count) != preset_boolean:
                relation = COUNT_COMPARE_TFQ_RELATION_DICT[relation]['contrapositive']

        # determine the contrapositive relation for the proposition
        contrapositive_relation = COUNT_COMPARE_TFQ_RELATION_DICT[relation]['contrapositive']

        # prepare the main proposition text
        base_prompt_text = (
            random.choice(COUNT_COMPARE_TFQ_RELATION_DICT[relation]['templates'])
            .replace('<OBJ1>', label1)
            .replace('<OBJ2>', label2)
        )
        prompt_caption = 'yes' if preset_boolean else 'no'
        base_prompt_suffix_text = random.choice(PROMPT_TFQ_HINT_TEMPLATES)

        # prepare the contrapositive proposition text
        cp_base_prompt_text = (
            random.choice(COUNT_COMPARE_TFQ_RELATION_DICT[contrapositive_relation]['templates'])
            .replace('<OBJ1>', label1)
            .replace('<OBJ2>', label2)
        )
        cp_prompt_caption = 'yes' if not preset_boolean else 'no'

        # build the chain-of-thought texts
        chain_of_thought_base_text = (
            COUNT_COMPARE_TFQ_CoT_CAPTION_TEMPLATE
            .replace('<OBJ1>', label1)
            .replace('<OBJ2>', label2)
            .replace('<OBJ1_COUNT>', str(label1_inst_count))
            .replace('<OBJ2_COUNT>', str(label2_inst_count))
            .replace('\n', ' ')
        )
        chain_of_thought_prompt_suffix_text = PROMPT_TFQ_CoT_HINT_TEMPLATE

        return {
            'meta': {
                'label1': {
                    'label': label1,
                    'ids': [inst.object_id for inst in label1_instances],
                    'count': label1_inst_count,
                },
                'label2': {
                    'label': label2,
                    'ids': [inst.object_id for inst in label2_instances],
                    'count': label2_inst_count,
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
                .replace('<RELATION>', COUNT_COMPARE_TFQ_RELATION_DICT[relation]['text'])
                .replace('<BOOLEAN>', '' if preset_boolean else 'not')
                .replace('<ANSWER>', prompt_caption)
            ),
            'ref_captions': CAPTION_TFQ_AFFIRMATIVES if preset_boolean else CAPTION_TFQ_NEGATIVES,

            'cp_prompt': cp_base_prompt_text + base_prompt_suffix_text,
            'cp_CoT_prompt': cp_base_prompt_text + chain_of_thought_prompt_suffix_text,
            'cp_caption': cp_prompt_caption,
            'cp_CoT_caption': (
                chain_of_thought_base_text
                .replace('<RELATION>', COUNT_COMPARE_TFQ_RELATION_DICT[contrapositive_relation]['text'])
                .replace('<BOOLEAN>', '' if not preset_boolean else 'not')
                .replace('<ANSWER>', cp_prompt_caption)
            ),
            'cp_ref_captions': CAPTION_TFQ_AFFIRMATIVES if not preset_boolean else CAPTION_TFQ_NEGATIVES,
        }
