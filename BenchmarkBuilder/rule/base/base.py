import random
import traceback
from abc import ABC, abstractmethod
from itertools import product
from typing import Optional, List, Dict, Any, Union

import click

from BenchmarkBuilder.utils.io import export_dict_as_json_file
from BenchmarkBuilder.utils.scene import SceneData, SceneInstance


class RuleBasedQGen(ABC):
    """Base class for rule-based question generator"""
    DEFAULT_EXCLUDED_OBJ_LABELS = ['wall', 'floor', 'ceiling', 'object', 'item']

    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str,
            question_type: str = 'DEFAULT_RULE',
            excluded_labels: Optional[List[str]] = None,
    ) -> None:
        """Initialize rule-based question generator"""
        self.scene_data = SceneData(scene_stat_json_file)
        self.scene_id = self.scene_data.scene_id

        # configure valid instances
        self.excluded_labels = set(excluded_labels or self.DEFAULT_EXCLUDED_OBJ_LABELS)

        # configure metadata and export JSON file
        self.question_type = question_type
        self.output_json_file = output_json_file

    # candidate pool preparation related methods
    @staticmethod
    def _custom_instance_filter(instance: SceneInstance) -> bool:
        """Override to implement custom filtering logic"""
        return True

    def _get_available_instances(self, allow_repeated: bool) -> List[SceneInstance]:
        """Get available instances for question generation"""
        return [
            instance
            for instance in self.scene_data.instances
            if (instance.label not in self.excluded_labels) and
               (self._custom_instance_filter(instance)) and
               (allow_repeated or len(self.scene_data.get_instances_by_label(instance.label)) == 1)
        ]

    @abstractmethod
    def _get_candidates(self) -> List[SceneInstance]:
        """Override to implement custom candidate selection logic"""
        raise NotImplementedError

    def _prepare_candidate_pool(
            self,
            allow_duplicate_candidates: bool = False,
            n_questions: int = 5
    ) -> List[Union[
        SceneInstance,
        tuple[SceneInstance, SceneInstance],
        tuple[tuple[SceneInstance, SceneInstance], tuple[SceneInstance, SceneInstance]],
        str,
        tuple[str, str],
        tuple[tuple[str, str], tuple[str, str]]
    ]]:
        """Prepare candidate pool for question generation"""
        candidates = self._get_candidates()
        if not allow_duplicate_candidates and len(candidates) < n_questions:
            click.echo(
                f'[WARN] {self.scene_id} - {self.question_type}: '
                f'Only {len(candidates)} unique candidates available for {n_questions} questions. '
                f'Sampling only {len(candidates)} questions.'
            )
            selected_candidates = random.sample(candidates, k=len(candidates))
        else:
            selected_candidates = (
                random.sample(candidates, k=n_questions)
                if not allow_duplicate_candidates else
                random.choices(candidates, k=n_questions)
            )
        return selected_candidates

    # question generation helper methods
    @abstractmethod
    def _form_question_dict(self, **kwargs) -> Dict[str, Any]:
        """Override to implement question dictionary formation logic"""
        raise NotImplementedError

    def _generate(
            self,
            candidate: Union[
                SceneInstance,
                tuple[SceneInstance, SceneInstance],
                tuple[tuple[SceneInstance, SceneInstance], tuple[SceneInstance, SceneInstance]]
            ],
            q_dicts: Union[List[Dict[str, Any]], List[None]],
            max_attempts: int,
            **kwargs
    ) -> Dict[str, Any]:
        """Generate question dictionary"""
        for attempt in range(max_attempts):
            try:
                q_dict = self._form_question_dict(candidate=candidate, **kwargs)
                if q_dict != {} and q_dict not in q_dicts:
                    return q_dict
                else:
                    click.echo(f'[WARN] {self.scene_id} - {self.question_type} - Attempt {attempt + 1}/{max_attempts}: '
                               f'Duplicate or empty question set generated. Retrying...')
            except Exception as e:
                tb = traceback.format_exc()
                click.echo(f'[ERROR] {self.scene_id} - {self.question_type} - Attempt {attempt + 1}/{max_attempts}: '
                           f'Failed to generate question set as \"{e}\".\nTraceback: \"{tb}\".\nRetrying...')
        click.echo(f'[ERROR] {self.scene_id} - {self.question_type} '
                   f'Failed to generate question set after {max_attempts} attempts. Skipping...')
        return {}

    def _export_question_dicts(self, question_dicts: List[Dict[str, Any]]) -> None:
        """Export question dictionaries to the output JSON file"""
        if not question_dicts:
            click.echo(f'[WARN] {self.scene_id} - {self.question_type}: No questions generated. Skipping...')
            return
        for q_dict in question_dicts:
            # skip empty question dictionaries
            if not q_dict:
                continue
            export_dict_as_json_file(
                {
                    'scene_id': self.scene_data.scene_id,
                    'question_type': self.question_type,
                    **{
                        k: v.replace(
                            '\n', ' ').replace(
                            '\\', '').replace(
                            '\t', ' ').replace(
                            '  ', ' ').strip()
                        if isinstance(v, str) else v
                        for k, v in q_dict.items()
                    }
                },
                self.output_json_file
            )


class SingleObjectCandidateMixin(RuleBasedQGen, ABC):
    """Mixin class for single object question generation"""

    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str,
            question_type: str = 'DEFAULT_RULE',
            excluded_labels: Optional[List[str]] = None,
            allow_repeated_objects: bool = True,
    ) -> None:
        """Initialize single object question generator"""
        super().__init__(scene_stat_json_file, output_json_file, question_type, excluded_labels)
        self.allow_repeated_objects = allow_repeated_objects

    def _get_candidates(self) -> List[SceneInstance]:
        """Get candidates for single object question generation"""
        return self._get_available_instances(self.allow_repeated_objects)


class DualObjectsCandidateMixin(RuleBasedQGen, ABC):
    """Mixin class for dual objects question generation"""

    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str,
            question_type: str = 'DEFAULT_RULE',
            excluded_labels: Optional[List[str]] = None,
            allow_repeated_inst1s: bool = True,
            allow_repeated_inst2s: bool = True,
    ) -> None:
        """Initialize dual objects question generator"""
        super().__init__(scene_stat_json_file, output_json_file, question_type, excluded_labels)
        self.allow_repeated_inst1s = allow_repeated_inst1s
        self.allow_repeated_inst2s = allow_repeated_inst2s

    def _get_candidates(self) -> List[tuple[SceneInstance, SceneInstance]]:
        """Get candidate instance pairs for dual objects question generation"""
        candidate_inst1s = self._get_available_instances(self.allow_repeated_inst1s)
        candidate_inst2s = self._get_available_instances(self.allow_repeated_inst2s)

        return [
            (inst1, inst2)
            for inst1, inst2 in product(candidate_inst1s, candidate_inst2s)
            if inst1 != inst2
        ]


class DualObjectPairsCandidateMixin(RuleBasedQGen, ABC):
    def __init__(
            self,
            scene_stat_json_file: str,
            output_json_file: str,
            question_type: str = 'DEFAULT_RULE',
            excluded_labels: Optional[List[str]] = None,
            allow_repeated_inst1as: bool = True,
            allow_repeated_inst1bs: bool = True,
            allow_repeated_inst2as: bool = True,
            allow_repeated_inst2bs: bool = True,
    ) -> None:
        """Initialize dual object pairs question generator"""
        super().__init__(scene_stat_json_file, output_json_file, question_type, excluded_labels)

        self.allow_repeated_inst1as = allow_repeated_inst1as
        self.allow_repeated_inst1bs = allow_repeated_inst1bs
        self.allow_repeated_inst2as = allow_repeated_inst2as
        self.allow_repeated_inst2bs = allow_repeated_inst2bs

    def _get_candidates(self) -> List[
        tuple[tuple[SceneInstance, SceneInstance], tuple[SceneInstance, SceneInstance]]
    ]:
        """Get candidate instance pairs for dual object pairs question generation"""
        candidate_inst1as = self._get_available_instances(self.allow_repeated_inst1as)
        candidate_inst1bs = self._get_available_instances(self.allow_repeated_inst1bs)
        candidate_inst2as = self._get_available_instances(self.allow_repeated_inst2as)
        candidate_inst2bs = self._get_available_instances(self.allow_repeated_inst2bs)

        return [
            ((obj1a, obj1b), (obj2a, obj2b))
            for (obj1a, obj1b), (obj2a, obj2b) in product(
                product(candidate_inst1as, candidate_inst1bs),
                product(candidate_inst2as, candidate_inst2bs)
            )
            # distinct elements in each pair
            if obj1a != obj1b and obj2a != obj2b
               # distinct pairs
               and {obj1a, obj1b} != {obj2a, obj2b}
        ]


class TFQMixin(RuleBasedQGen, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_type = 'TFQ'

    @staticmethod
    def _get_preset_booleans(
            n_questions: int,
            enforce_balanced: bool = True,
    ) -> List[bool]:
        """Get preset boolean values for question dictionary"""
        if enforce_balanced:
            return [i % 2 == 0 for i in range(n_questions)]
        else:
            return [random.choice([True, False]) for _ in range(n_questions)]

    def generate(
            self,
            n_questions: int,
            max_attempts: int = 5,
            enforce_balanced: bool = True,
            allow_duplicate_objects: bool = False,
    ) -> None:
        """Generate True/False questions"""
        preset_booleans = self._get_preset_booleans(n_questions, enforce_balanced)
        candidates = self._prepare_candidate_pool(allow_duplicate_objects, n_questions)
        q_dicts = []
        for candidate, preset_boolean in zip(candidates, preset_booleans):
            q_dicts.append(
                self._generate(
                    candidate=candidate,
                    q_dicts=q_dicts,
                    max_attempts=max_attempts, preset_boolean=preset_boolean
                )
            )
        self._export_question_dicts(q_dicts)


class SAQMixin(RuleBasedQGen, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_type = 'NUM-SAQ'

    def generate(
            self,
            n_questions: int,
            max_attempts: int = 5,
            allow_duplicate_instances: bool = False,
    ) -> None:
        """Generate Short Answer Questions"""
        candidates = self._prepare_candidate_pool(allow_duplicate_instances, n_questions)
        q_dicts = []
        for candidate in candidates:
            q_dicts.append(
                self._generate(
                    candidate=candidate,
                    q_dicts=q_dicts,
                    max_attempts=max_attempts
                )
            )
        self._export_question_dicts(q_dicts)
