from abc import ABC

from BenchmarkBuilder.utils.scene import SceneData


class RuleBasedQuestionGenerator(ABC):
    def __init__(
            self,
            scene_stat_json_file: str
    ) -> None:
        self.scene_data = SceneData(scene_stat_json_file)
        
