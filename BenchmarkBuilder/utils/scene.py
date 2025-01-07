from dataclasses import dataclass
from typing import List

from BenchmarkBuilder.utils.io import load_json_file_as_dict


@dataclass
class SceneInstance:
    object_id: str
    label: str
    center: List[float]
    bbox_xyz_min: List[float]
    bbox_xyz_max: List[float]
    bbox_xyz_len: List[float]
    bbox_volume: float


class SceneData:
    def __init__(
            self,
            scene_stat_json_file: str
    ):
        self.scene_stat_dict = load_json_file_as_dict(scene_stat_json_file)
        self.instances = [SceneInstance(**inst) for inst in self.scene_stat_dict['instances']]
        self.pairwise_distances = self.scene_stat_dict['pairwise_distances']

    def _index_by_label(self) -> dict[str, List[SceneInstance]]:
        """Map the instances by label"""
        label_map = {}
        for inst in self.instances:
            if inst.label not in label_map:
                label_map[inst.label] = []
            label_map[inst.label].append(inst)
        return label_map
