import random
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Optional, List

import click

from BenchmarkBuilder.utils.io import export_dict_as_json_file
from BenchmarkBuilder.utils.scene import SceneData, SceneInstance


class RuleBasedQuestionGenerator(ABC):
    """Base class for rule-based question generator"""
    DEFAULT_EXCLUDED_OBJ_LABELS = {'wall', 'floor', 'ceiling', 'object'}

    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str,
            excluded_labels: Optional[List[str]] = None,
            require_singleton_objs: bool = False,
    ) -> None:
        """Initialize rule-based question generator"""
        self.scene_data = SceneData(scene_stat_json_file)

        # configure valid instances
        self.excluded_labels = set(excluded_labels or self.DEFAULT_EXCLUDED_OBJ_LABELS)
        self.require_singleton_objs = require_singleton_objs

        # configure export JSON file
        self.export_json_file = export_json_file

    @staticmethod
    def _custom_instance_filter(instance: SceneInstance) -> bool:
        """Override to implement custom filtering logic"""
        return True

    def _get_filtered_instances(self) -> List[SceneInstance]:
        """Get filtered instances based on all criteria"""
        filtered = [
            inst for inst in self.scene_data.instances
            if inst.label not in self.excluded_labels and
               self._custom_instance_filter(inst)
        ]

        # filter 
        if self.require_singleton_objs:
            unique_filtered = []
            for label in self.scene_data.unique_labels:
                instances = [inst for inst in filtered if inst.label == label]
                if len(instances) == 1:
                    unique_filtered.extend(instances)
            return unique_filtered

        return filtered

    def _get_available_object_ids(self) -> list[str]:
        """Get available object IDs meeting all criteria"""
        return [inst.object_id for inst in self._get_filtered_instances()]

    def _get_available_labels(self) -> list[str]:
        """Get available labels meeting all criteria"""
        return list({inst.label for inst in self._get_filtered_instances()})

    @staticmethod
    def _validate_args(n_questions: int, max_retries: int) -> None:
        """Validate input arguments"""
        if n_questions < 1:
            raise ValueError(f'Invalid number of questions: {n_questions}')
        if max_retries < n_questions:
            raise ValueError(f'max_retries must be >= n_questions')

    @abstractmethod
    def generate(self, n_questions: int, max_retries: int, **kwargs) -> None:
        """Generate questions based on the scene data"""
        self._validate_args(n_questions, max_retries)

    def _export_question_dicts(
            self,
            question_dicts: list[dict],
    ) -> None:
        """Export the question to the JSON file"""
        for q_dict in question_dicts:
            export_dict_as_json_file(q_dict, self.export_json_file)


class ObjectAttributeQuestionGenerator(RuleBasedQuestionGenerator):
    """Base class for rule-based short answer question generator"""

    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str,
            excluded_labels: list[str] | None = None,
            require_singleton_objs: bool = False,
    ) -> None:
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            excluded_labels,
            require_singleton_objs
        )

    def generate(
            self,
            n_questions: int,
            max_retries: int = 20,
            enforce_unique_labels: bool = False,
            **kwargs,
    ) -> None:
        self._validate_args(n_questions, max_retries)
        available_labels = self._get_available_labels()
        if not available_labels:
            click.echo('[WARN] No available labels found for generating questions, '
                       'NO questions will be generated.')
            return

        question_dicts = []
        used_labels = set()
        remaining_labels = set(available_labels)

        retries = 0
        while len(question_dicts) < n_questions and retries < max_retries:
            if enforce_unique_labels and not remaining_labels:
                break

            label = random.choice(list(remaining_labels if enforce_unique_labels else available_labels))
            if enforce_unique_labels:
                remaining_labels.remove(label)

            question_dict = self._form_question_dict(label)
            if question_dict not in question_dicts:
                question_dicts.append(question_dict)
            else:
                click.echo(f'[WARN] Duplicate question generated for label: {label}, skipping...')
            used_labels.add(label)
            retries += 1

        if len(question_dicts) < n_questions:
            click.echo(f'[WARN] Generated only {len(question_dicts)} / {n_questions} questions '
                       f'as all {len(available_labels)} available labels are used '
                       f'before reaching target question count.\n'
                       f'Consider disabling strict unique mode or increasing the number of retries.')
        self._export_question_dicts(question_dicts)

    @abstractmethod
    def _form_question_dict(self, label: str) -> dict:
        """Form question for counting objects in the scene"""
        pass


class ObjectPairwiseDistanceQuestionGenerator(RuleBasedQuestionGenerator):
    """Base class for rule-based dual object question generator"""

    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str,
            excluded_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            scene_stat_json_file,
            export_json_file,
            excluded_labels,
            require_singleton_objs=True
        )
        self._pairwise_distance_dict = self._enum_valid_pairwise_distance_dict()

    def generate(
            self,
            n_questions: int,
            max_retries: int = 20,
            enforce_unique_labels: bool = False,
            **kwargs,
    ) -> None:
        pass

    def _enum_valid_pairwise_distance_dict(self) -> dict[tuple[str, str], float]:
        """Enumerate the valid pairwise distances between objects"""
        sole_obj_ids = self._get_available_object_ids()
        pairwise_distance_dict = {
            (obj_id1, obj_id2): self.scene_data.get_pairwise_distance(obj_id1, obj_id2)
            for obj_id1, obj_id2 in combinations(sole_obj_ids, 2)
        }
        return pairwise_distance_dict

    @abstractmethod
    def _form_question_dict(self, **kwargs) -> dict:
        """Form question for counting objects in the scene"""
        pass
