import os.path
from abc import ABC, abstractmethod

from BenchmarkBuilder.utils.io import export_dict_as_json_file
from BenchmarkBuilder.utils.scene import SceneData


class RuleBasedQuestionGenerator(ABC):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str,
    ) -> None:
        self.scene_data = SceneData(scene_stat_json_file)

        # configure export JSON file
        self.export_json_file = os.path.abspath(export_json_file)

    def generate(
            self,
            n_questions: int = 5,
            max_iters: int = 20,
    ) -> None:
        """Generate questions based on the scene data"""
        if max_iters < n_questions:
            raise ValueError(f'max_iters {max_iters} should be no less than n_questions {n_questions}')

        question_dict_set = set()
        for _ in range(max_iters):
            question_dict = self._generate_question()
            question_dict_set.add(frozenset(question_dict.items()))
            if len(question_dict_set) >= n_questions:
                break
        
        # raise warning if not enough questions are generated
        if len(question_dict_set) < n_questions:
            print(f'Warning: Only {len(question_dict_set)} out of {n_questions} questions are generated')

        # export questions to JSON file
        for question_dict in question_dict_set:
            export_dict_as_json_file(dict(question_dict), self.export_json_file)

    @abstractmethod
    def _generate_question(self) -> dict[str, str | int | float]:
        """Generate a question based on the scene data"""
        raise NotImplementedError
