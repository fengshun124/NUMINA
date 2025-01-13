import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from BenchmarkBuilder.rule.base import ObjectAttributeQuestionGenerator

PROMPT_QUESTION_TEMPLATES = [
    'Can you count the number of <OBJ> in the room? ',
    'Can you count the <OBJ> in the room? ',
    'Can you tell me how many <OBJ> there are in the room? ',
    'Can you tell me the number of <OBJ> in the room? ',
    'Count the number of <OBJ> in the room. ',
    'How many <OBJ> can you see in the room? ',
    'How many <OBJ> are there in the room? ',
    'How many <OBJ> are in the room? ',
    'Please count the number of <OBJ> in the room. ',
    'Please provide the number of <OBJ> in the room. ',
    'Please tell me the number of <OBJ> in the room. ',
    'What is the number of <OBJ> in the room? ',
    'What is the count of <OBJ> in the room? ',
]


class CountSAQGenerator(ObjectAttributeQuestionGenerator):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str = './output/NUM-count-SAQ.json',
            excluded_labels: list[str] | None = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            excluded_labels,
            require_singleton_objs=False
        )

    def _form_question_dict(self, label: str) -> dict:
        """Form question for counting objects in the scene"""
        instances = self.scene_data.get_instances_by_label(label)
        count = len(instances)

        return {
            'scene_id': self.scene_data.scene_id,
            'count_meta': {
                'label': label,
                'obj_ids': [inst.object_id for inst in instances],
            },
            'prompt': random.choice(PROMPT_QUESTION_TEMPLATES).replace('<OBJ>', label),
            'caption': str(count),
            'CoT_caption': f'<<answer:{count}>>',
            'question_type': 'Rule-ShortAnswer'
        }
