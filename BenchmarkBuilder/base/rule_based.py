import os.path
from abc import ABC, abstractmethod

from BenchmarkBuilder.utils.io import export_dict_as_json_file
from BenchmarkBuilder.utils.scene import SceneData


class RuleBasedQuestionGenerator(ABC):
    def __init__(
            self,
            scene_stat_json_file: str,
            export_json_file: str,
            exclude_labels: list[str] = None,
    ) -> None:
        self.scene_data = SceneData(scene_stat_json_file)

        # configure export JSON file
        self.export_json_file = os.path.abspath(export_json_file)

        # configure excluded object labels
        self.exclude_labels = exclude_labels or ['wall', 'floor', 'ceiling', 'object']

    @abstractmethod
    def generate(self, **kwargs) -> None:
        """Generate questions based on the scene data"""
        pass

    def _export_question(self, question_dict: dict[str, str | int | float]) -> None:
        """Export the question to the JSON file"""
        export_dict_as_json_file(question_dict, self.export_json_file)
