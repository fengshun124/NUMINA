import random

from BenchmarkBuilder.rule.base import (
    PROMPT_SAQ_HINT_TEMPLATES,
    SingleLabelBasedQuestionGenerator,
)

COUNT_SAQ_TEMPLATES = [
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

COUNT_TFQ_RELATION_DICT = {
    '>': {
        'func': lambda x, y: x > y,
        'text': 'greater than',
        'templates': [
            'Is the number of <OBJ1> greater than the number of <OBJ2>? ',
            'Are there more <OBJ1> than <OBJ2> in the room? '
        ]
    },
    '<': {
        'func': lambda x, y: x < y,
        'text': 'less than',
        'templates': [
            'Is the number of <OBJ1> less than the number of <OBJ2>? ',
            'Are there fewer <OBJ1> than <OBJ2> in the room? '
        ]
    },
    '=': {
        'func': lambda x, y: x == y,
        'text': 'equal to',
        'templates': [
            'Is the number of <OBJ1> equal to the number of <OBJ2>? ',
            'Are there the same number of <OBJ1> and <OBJ2> in the room? '
        ]
    }
}

COUNT_TFQ_CoT_TEMPLATE = """Given the number of <OBJ1> as <OBJ1_COUNT> and the number of <OBJ2> as <OBJ2_COUNT>,
the number of <OBJ1> is <OBJ1_RULE> the number of <OBJ2>.
Therefore, the answer is <<answer:<ANSWER>>>."""


class CountSAQGenerator(SingleLabelBasedQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str = './output/NUM-count-SAQ.json',
            excluded_labels: list[str] | None = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            excluded_labels=excluded_labels,
            allow_repeated_objs=True,
            question_type='RULE-count-SAQ',
        )

    def _form_question_dict(self, label: str) -> dict:
        """Form question for counting objects in the scene"""
        instances = self.scene_data.get_instances_by_label(label)
        obj_count = len(instances)

        return {
            'meta': {
                'label': label,
                'obj_ids': [inst.object_id for inst in instances],
            },
            'prompt': random.choice(COUNT_SAQ_TEMPLATES).replace(
                '<OBJ>', label) + random.choice(PROMPT_SAQ_HINT_TEMPLATES),
            'caption': str(obj_count),
            'CoT_caption': f'<<answer:{obj_count}>>',
            'ref_captions': [
                f'{obj_count}',
                f'{obj_count} {label}',
            ],
        }


class CountTFQGenerator(SingleLabelBasedQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str = './output/NUM-count-TFQ.json',
            excluded_labels: list[str] | None = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            excluded_labels=excluded_labels,
            allow_repeated_label1=True,
            allow_repeated_label2=True,
            question_type='RULE-count-TFQ',
        )

    def _form_question_dict(
            self, label1: str, label2: str, preset_boolean: bool
    ) -> dict:
        """Form question for comparing the count of two objects in the scene"""
        instances1 = self.scene_data.get_instances_by_label(label1)
        instances2 = self.scene_data.get_instances_by_label(label2)
        obj1_count, obj2_count = len(instances1), len(instances2)

        # select the rule for comparison
        relation = random.choice(list(COUNT_TFQ_RELATION_DICT.keys()))

        return {
            'meta': {
                'label1': label1,
                'label2': label2,
                'obj1_ids': [inst.object_id for inst in instances1],
                'obj2_ids': [inst.object_id for inst in instances2],
            },
            'prompt': random.choice(COUNT_TFQ_RELATION_DICT[relation]['templates']).replace(
                '<OBJ1>', label1).replace(
                '<OBJ2>', label2)
                      + random.choice(PROMPT_SAQ_HINT_TEMPLATES),
            'caption': f'{label1} {COUNT_TFQ_RELATION_DICT[relation]["text"]} {label2}',
            'CoT_caption': COUNT_TFQ_CoT_TEMPLATE.replace(
                '<OBJ1>', label1).replace(
                '<OBJ2>', label2).replace(
                '<OBJ1_COUNT>', str(obj1_count)).replace(
                '<OBJ2_COUNT>', str(obj2_count)).replace(
                '<OBJ1_RULE>', COUNT_TFQ_RELATION_DICT[relation]['text']).replace(
                '<ANSWER>', str(COUNT_TFQ_RELATION_DICT[relation]['func'](obj1_count, obj2_count))),
            'ref_captions': [
                f'{label1} {COUNT_TFQ_RELATION_DICT[relation]["text"]} {label2}',
                f'{obj1_count} {label1} {COUNT_TFQ_RELATION_DICT[relation]["text"]} {obj2_count} {label2}',
            ],
        }
