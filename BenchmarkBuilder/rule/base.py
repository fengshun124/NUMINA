import random
from abc import ABC, abstractmethod
from typing import Optional, List

import click

from BenchmarkBuilder.utils.io import export_dict_as_json_file
from BenchmarkBuilder.utils.scene import SceneData, SceneInstance

PROMPT_SAQ_HINT_TEMPLATES = [
    'Kindly provide a number as the answer. ',
    'Give a number as the answer. ',
    'Give a numerical response. ',
    'Offer a number as the answer. ',
    'Provide a number as the answer. ',
    'Please provide a numerical response. ',
    'Please provide a number as the answer. ',
    'Please reply with a number only. ',
    'Reply with a number only. ',
]

PROMPT_TFQ_HINT_TEMPLATES = [
    'Kindly provide a "yes" or "no" as the answer. ',
    'Give a "yes" or "no" as the answer. ',
    'Select "yes" or "no" as the answer. ',
    'Offer a "yes" or "no" as the answer. ',
    'Provide a "yes" or "no" as the answer. ',
    'Please provide a "yes" or "no" as the answer. ',
    'Please reply with a "yes" or "no" only. ',
    'Reply with a "yes" or "no" only. ',
]


class RuleBasedQuestionGenerator(ABC):
    """Base class for rule-based question generator"""
    DEFAULT_EXCLUDED_OBJ_LABELS = ['wall', 'floor', 'ceiling', 'object']

    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str,
            question_type: str = 'DEFAULT_RULE',
            excluded_labels: Optional[List[str]] = None,
            allow_repeated_objs: Optional[bool] = True,
    ) -> None:
        """Initialize rule-based question generator"""
        self.scene_data = SceneData(scene_stat_json_file)
        self.scene_id = self.scene_data.scene_id

        # configure valid instances
        self.excluded_labels = set(excluded_labels or self.DEFAULT_EXCLUDED_OBJ_LABELS)
        self.allow_repeated_objs = allow_repeated_objs

        # configure metadata and export JSON file
        self.question_type = question_type
        self.output_json_file = output_json_file

    @staticmethod
    def _custom_instance_filter(instance: SceneInstance) -> bool:
        """Override to implement custom filtering logic"""
        return True

    @property
    def _unique_instances(self) -> list[SceneInstance]:
        """Get unique instances that appear only once in the scene"""
        return [
            inst for inst in self.scene_data.instances
            if inst.label not in self.excluded_labels and
               self._custom_instance_filter(inst) and
               (len(self.scene_data.get_instances_by_label(inst.label)) == 1)
        ]

    @property
    def _unique_labels(self) -> list[str]:
        """Get labels of unique instances"""
        return [inst.label for inst in self._unique_instances]

    @property
    def _repeated_instances(self) -> list[SceneInstance]:
        """Get instances that repeatedly appear in the scene"""
        return [
            inst for inst in self.scene_data.instances
            if inst.label not in self.excluded_labels and
               self._custom_instance_filter(inst) and
               (len(self.scene_data.get_instances_by_label(inst.label)) > 1)
        ]

    @property
    def _repeated_labels(self) -> list[str]:
        """Get labels of repeated instances"""
        return [inst.label for inst in self._repeated_instances]

    @property
    def _available_instances(self) -> list[SceneInstance]:
        """Get available instances meeting all criteria"""
        if isinstance(self.allow_repeated_objs, bool) and self.allow_repeated_objs:
            return self._unique_instances + self._repeated_instances
        elif not self.allow_repeated_objs:
            return self._unique_instances
        else:
            click.exceptions.UsageError(f'[ERROR] Invalid value for allow_repeated_objs: '
                                        f'{self.allow_repeated_objs}')

    @property
    def _available_inst_labels(self) -> list[str]:
        """Get available labels meeting all criteria"""
        return [inst.label for inst in self._available_instances]

    @property
    def _available_inst_obj_ids(self) -> list[str]:
        """Get available object IDs meeting all criteria"""
        return [inst.object_id for inst in self._available_instances]

    def _export_question_dicts(self, question_dicts: list[dict]) -> None:
        """Export question dictionaries to the output JSON file"""
        for q_dict in question_dicts:
            export_dict_as_json_file(
                {
                    'scene_id': self.scene_data.scene_id,
                    'question_type': self.question_type,
                    **{k: v.replace('\n', ' ').replace('\\', '').strip()
                    if isinstance(v, str) else
                    v for k, v in q_dict.items()}
                },
                self.output_json_file
            )

    @abstractmethod
    def generate(self, **kwargs) -> None:
        """Generate questions based on the rule"""
        raise NotImplementedError


class TFQGeneratorMixin(ABC):
    """Mixin class for True/False question generator"""

    @staticmethod
    def _generate_preset_booleans(
            n_questions: int,
            enforce_balanced_boolean: bool = True
    ) -> list[bool]:
        """Generate preset booleans for True/False questions"""
        return (
            [idx % 2 == 0 for idx in range(n_questions)]
            if enforce_balanced_boolean else
            [random.choice([True, False]) for _ in range(n_questions)]
        )


class SingleLabelBasedQuestionGenerator(RuleBasedQuestionGenerator, ABC):
    """Base class for single-label based question generator"""

    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str,
            question_type: str,
            excluded_labels: Optional[List[str]] = None,
            allow_repeated_objs: bool = False,
    ) -> None:
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            question_type=question_type,
            excluded_labels=excluded_labels,
            allow_repeated_objs=allow_repeated_objs,
        )

    def _prepare_label_pool(
            self, n_questions: int, allow_duplicate: bool = True
    ) -> tuple[list[str], int]:
        # validate input arguments
        if n_questions < 1:
            raise ValueError(f'[ERROR] Invalid number of questions: {n_questions}, '
                             f'expected integer >= 1')
        # get available labels
        available_labels = self._available_inst_labels
        if len(available_labels) < 1:
            raise ValueError(f'[ERROR] No instances found for generating questions, '
                             f'possibly due to invalid filter criteria for scene: {self.scene_data.scene_id}, '
                             f'aborting question generation...')

        # adjust question count if no sufficient items are available
        label_pool = (
            random.choices(available_labels, k=n_questions)
            if allow_duplicate else
            random.sample(available_labels, min(n_questions, len(available_labels)))
        )
        if len(label_pool) < n_questions:
            click.echo(f'[WARN] Insufficient instances found for generating questions, '
                       f'only {len(label_pool)} out of {len(self.scene_data.instances)} '
                       f'available for scene: {self.scene_data.scene_id}')

        return label_pool, n_questions

    @abstractmethod
    def _form_question_dict(self, label: str) -> dict:
        """Form question for counting objects in the scene"""
        raise NotImplementedError

    def generate(self, **kwargs) -> None:
        """Generate questions based on the rule"""
        n_questions = kwargs.get('n_questions')
        max_retries = kwargs.get('max_retries', 3)
        allow_duplicate = kwargs.get('allow_duplicate', True)

        try:
            label_pool, n_questions = self._prepare_label_pool(n_questions, allow_duplicate)
        except Exception as e:
            click.echo(e)
            return

        question_dicts = []
        for label in label_pool:
            for attempt in range(1, max_retries + 1):
                try:
                    q_dict = self._form_question_dict(label)
                    if q_dict not in question_dicts or not q_dict:
                        question_dicts.append(q_dict)
                        break
                    click.echo(f'[WARN] ({attempt}/{max_retries}) Duplicate {self.question_type} generated '
                               f'for {self.scene_id}-\"{label}\", retrying...')
                except Exception as e:
                    click.echo(f'[ERROR] ({attempt}/{max_retries}) Failed to generate {self.question_type} '
                               f'for {self.scene_id}-\"{label}\": \"{e}\" retrying...')
            else:
                click.echo(f'[ERROR] {max_retries} attempts failed to generate {self.question_type} '
                           f'for {self.scene_id}-\"{label}\", skipping label...')

        self._export_question_dicts(question_dicts)


class DualLabelsBasedQuestionGenerator(RuleBasedQuestionGenerator, ABC):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str,
            question_type: str,
            excluded_labels: Optional[List[str]] = None,
            allow_repeated_label1: bool = False,
            allow_repeated_label2: bool = False,
    ):
        super().__init__(
            scene_stat_json_file=scene_stat_json_file,
            output_json_file=output_json_file,
            question_type=question_type,
            excluded_labels=excluded_labels,
            allow_repeated_objs=True,
        )
        self.allow_repeated_label1 = allow_repeated_label1
        self.allow_repeated_label2 = allow_repeated_label2

    @abstractmethod
    def _form_question_dict(self, label1: str, label2: str) -> dict:
        """Form question for comparing the count of two objects in the scene"""
        raise NotImplementedError

    def _prepare_label_pair_pool(
            self,
            n_questions: int,
            allow_duplicate: Optional[bool] = True
    ) -> tuple[list[tuple[str, str]], int]:
        # validate input arguments
        if n_questions < 1:
            raise ValueError(f'[ERROR] Invalid number of {self.question_type}: {n_questions}, '
                             f'expected integer >= 1')
        # get available labels
        available_label1s = self._available_inst_labels if self.allow_repeated_label1 else self._unique_labels
        available_label2s = self._available_inst_labels if self.allow_repeated_label2 else self._unique_labels
        available_pairs = [(l1, l2) for l1 in available_label1s for l2 in available_label2s if l1 != l2]
        if len(available_pairs) < 1:
            raise ValueError(f'[ERROR] No instances found for generating questions, '
                             f'possibly due to invalid filter criteria for scene: {self.scene_data.scene_id}, '
                             f'aborting question generation...')

        # adjust question count if no sufficient items are available
        pair_pool = (
            random.choices(available_pairs, k=n_questions)
            if allow_duplicate else
            random.sample(available_pairs, min(n_questions, len(available_pairs)))
        )
        if len(pair_pool) < n_questions:
            click.echo(f'[WARN] Insufficient instances found for generating questions, '
                       f'only {len(pair_pool)} out of {len(available_pairs)} '
                       f'available for scene: {self.scene_data.scene_id}')

        return pair_pool, n_questions

    def generate(self, **kwargs) -> None:
        """Generate questions based on the rule"""
        n_questions = kwargs.get('n_questions')
        max_retries = kwargs.get('max_retries', 3)

        try:
            label_pair_pool, n_questions = self._prepare_label_pair_pool(n_questions)
        except Exception as e:
            click.echo(e)
            return

        question_dicts = []
        for label1, label2 in label_pair_pool:
            for attempt in range(1, max_retries + 1):
                try:
                    q_dict = self._form_question_dict(label1, label2)
                    if q_dict not in question_dicts or not q_dict:
                        question_dicts.append(q_dict)
                        break
                    click.echo(f'[WARN] ({attempt}/{max_retries}) Duplicate {self.question_type} generated '
                               f'for {self.scene_id}-\"{label1}/{label2}\", retrying...')
                except Exception as e:
                    click.echo(f'[ERROR] ({attempt}/{max_retries}) Failed to generate {self.question_type} '
                               f'for {self.scene_id}-\"{label1}/{label2}\" as \"{e}\", retrying...')
            else:
                click.echo(f'[ERROR] {max_retries} attempts failed to generate {self.question_type} '
                           f'for {self.scene_id}-{label1}-{label2}, skipping label pair...')

        self._export_question_dicts(question_dicts)
