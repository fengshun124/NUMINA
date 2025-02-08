from dataclasses import dataclass
from typing import List, Optional, Set

from BenchmarkBuilder.utils.io import load_json_file_as_dict


@dataclass
class SceneInstance:
    """Single object instance in a scene with its geometric properties"""
    object_id: str
    label: str
    center: List[float]
    bbox_xyz_min: List[float]
    bbox_xyz_max: List[float]
    bbox_xyz_len: List[float]
    bbox_volume: float

    def to_dict(self):
        """Export instance data as dictionary"""
        return {
            'object_id': self.object_id,
            'label': self.label,
            'center': self.center,
            'bbox_xyz_min': self.bbox_xyz_min,
            'bbox_xyz_max': self.bbox_xyz_max,
            'bbox_xyz_len': self.bbox_xyz_len,
            'bbox_volume': self.bbox_volume,
        }

    def __repr__(self):
        return f'SceneInstance({self.object_id}, {self.label})'

    def __hash__(self):
        # hash by object ID & label
        return hash((self.object_id, self.label))

    def __eq__(self, other):
        if not isinstance(other, SceneInstance):
            return False
        # check both ID and name
        return self.object_id == other.object_id and self.label == other.label


class SceneData:
    """Manages scene data and provides methods to query object relationships"""

    def __init__(self, scene_stat_json_file: str) -> None:
        """Initialize scene data from statistics JSON file"""
        self.scene_stat_dict = load_json_file_as_dict(scene_stat_json_file)[0]
        self.scene_id: str = self.scene_stat_dict['scene_id']
        self.instances = [SceneInstance(**inst) for inst in self.scene_stat_dict['instances']]

        self._instance_map = self.scene_stat_dict['instance_map']
        self._object_map = self.scene_stat_dict['object_map']
        self._pairwise_distances = self.scene_stat_dict['pairwise_distances']

    @property
    def unique_labels(self) -> List[str]:
        """Get list of unique object labels in scene"""
        return self.scene_stat_dict['unique_labels']

    @property
    def object_ids(self) -> List[str]:
        """Get list of object IDs in scene"""
        return list(self._object_map.keys())

    def get_instances_by_label(self, label: str) -> List[SceneInstance]:
        """Get all instances with given label"""
        if label not in self._instance_map:
            raise ValueError(f'Label "{label}" not found in scene')
        return [inst for inst in self.instances if inst.label == label]

    def get_instance_by_object_id(self, object_id: str | int) -> Optional[SceneInstance]:
        """Get instance with given object ID"""
        object_id = self._validate_obj_id_format(object_id)
        if object_id not in self._object_map:
            raise ValueError(f'Object ID "{object_id}" not found in scene')
        return next((inst for inst in self.instances if inst.object_id == int(object_id)), None)

    def get_pairwise_distance(self, obj_id1: str | int, obj_id2: str | int) -> float:
        """Get distance between two objects"""
        obj_id1 = self._validate_obj_id_format(obj_id1)
        obj_id2 = self._validate_obj_id_format(obj_id2)
        key = self._validate_distance_key(obj_id1, obj_id2)
        if key not in self._pairwise_distances:
            raise ValueError(f'No distance found between objects {obj_id1} and {obj_id2}')
        return self._pairwise_distances[key]

    def get_obj_surroundings(
            self,
            obj_id: str | int, radius: float,
            exclude_obj_ids: Optional[List[str | int]] = None,
            exclude_labels: Optional[List[str]] = None
    ) -> List[str]:
        """Get object IDs within radius distance of given object"""
        obj_id = self._validate_obj_id_format(obj_id)
        exclude_obj_ids_set = {self._validate_obj_id_format(e) for e in (exclude_obj_ids or [])}
        exclude_labels_set = set(exclude_labels or [])

        self._validate_obj_id(obj_id, exclude_obj_ids_set, exclude_labels_set)

        surrounding_objs = []
        for key, distance in self._pairwise_distances.items():
            obj1, obj2 = key.split('-')
            neighbor_id = obj2 if obj_id == obj1 else obj1

            if (obj_id in (obj1, obj2) and
                    distance <= radius and
                    neighbor_id not in exclude_obj_ids_set and
                    self._object_map[neighbor_id] not in exclude_labels_set):
                surrounding_objs.append(neighbor_id)

        return surrounding_objs

    def get_obj_k_neighbors(
            self,
            obj_id: str | int, k: int,
            exclude_obj_ids: Optional[List[str | int]] = None,
            exclude_labels: Optional[List[str]] = None
    ) -> List[str]:
        """Get k nearest neighboring object IDs to given object"""
        if k < 1:
            raise ValueError('k must be positive')

        obj_id = self._validate_obj_id_format(obj_id)
        exclude_obj_ids_set = {self._validate_obj_id_format(e) for e in (exclude_obj_ids or [])}
        exclude_labels_set = set(exclude_labels or [])

        self._validate_obj_id(obj_id, exclude_obj_ids_set, exclude_labels_set)

        neighbors = []
        for key, _ in sorted(self._pairwise_distances.items(), key=lambda x: x[1]):
            obj1, obj2 = key.split('-')
            neighbor_id = obj2 if obj_id == obj1 else obj1

            if (obj_id in (obj1, obj2) and
                    neighbor_id not in exclude_obj_ids_set and
                    self._object_map[neighbor_id] not in exclude_labels_set):
                neighbors.append(neighbor_id)
                if len(neighbors) == k:
                    break

        return neighbors

    # helper methods
    def _validate_obj_id(
            self, obj_id: str,
            exclude_obj_ids: Set[str], exclude_labels: Set[str]
    ) -> None:
        """Validate object ID exists and is not excluded"""
        if obj_id not in self._object_map:
            raise ValueError(f'Object ID "{obj_id}" not found in scene')
        if obj_id in exclude_obj_ids:
            raise ValueError(f'Target object ID "{obj_id}" is in exclude_obj_ids')
        if self._object_map[obj_id] in exclude_labels:
            raise ValueError(f'Target object label "{self._object_map[obj_id]}" is in exclude_labels')

    @staticmethod
    def _validate_obj_id_format(obj_id: str | int) -> str:
        """Validate and convert object ID to string format"""
        obj_id_str = str(obj_id)
        if not obj_id_str.isdigit():
            raise ValueError(f'Object ID "{obj_id}" is not a non-negative integer')
        return obj_id_str

    def _validate_distance_key(self, obj_id1: str | int, obj_id2: str | int) -> str:
        """Validate and format distance key"""
        if obj_id1 == obj_id2:
            raise ValueError('Object IDs must be different')
        if obj_id1 not in self._object_map:
            raise ValueError(f'Object ID "{obj_id1}" not found in scene')
        if obj_id2 not in self._object_map:
            raise ValueError(f'Object ID "{obj_id2}" not found in scene')
        return f'{min(int(obj_id1), int(obj_id2))}-{max(int(obj_id1), int(obj_id2))}'
