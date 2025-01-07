import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import click
import numpy as np
from joblib import delayed
from plyfile import PlyData
from scipy.spatial import distance, ConvexHull

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import export_dict_as_json_file
from utils.parallel import ParallelTqdm


@dataclass
class SceneInstanceMetric:
    object_id: str
    label: str
    points: np.ndarray

    obj_center: Optional[List[float]] = None

    # bounding box metrics
    bbox_xyz_min: Optional[List[float]] = None
    bbox_xyz_max: Optional[List[float]] = None
    bbox_xyz_len: Optional[List[float]] = None
    bbox_volume: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.points, np.ndarray):
            raise ValueError('Invalid point cloud data type. Expected numpy array')
        if self.points.shape[1] != 3:
            raise ValueError('Invalid point cloud data shape. Expected (n, 3)')

    def calc_metrics(self) -> None:
        """Calculate the object center and bounding box size"""
        try:
            if len(self.points) == 0:
                raise ValueError('Empty point cloud data')

            self.obj_center = np.mean(self.points, axis=0).tolist()
            # bbox size refers to the diagonal length of the bounding box
            self.bbox_xyz_min = np.min(self.points, axis=0).tolist()
            self.bbox_xyz_max = np.max(self.points, axis=0).tolist()
            self.bbox_xyz_len = np.ptp(self.points, axis=0).tolist()
            # calculate the volume of the bounding box (length x width x height)
            self.bbox_volume = float(np.prod(self.bbox_xyz_len))
        except Exception as e:
            print(f'Error calculating metrics for object {self.object_id}: {e}')


class ScanNetSceneAnalyzer:
    def __init__(self, scene_dir: str) -> None:
        self.scene_dir = Path(scene_dir).absolute()
        self.scene_id = self.scene_dir.name
        # validate if the scene directory exists
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f'Invalid scene directory path: {self.scene_dir}')
        self._init_scene_files()

    def _init_scene_files(self) -> None:
        self.ply_path = self.scene_dir / f'{self.scene_id}_vh_clean_2.ply'
        self.seg_json_path = self.scene_dir / f'{self.scene_id}_vh_clean_2.0.010000.segs.json'
        # self.agg_json_path = self.scene_dir / f'{self.scene_id}_vh_clean.aggregation.json'
        self.agg_json_path = self.scene_dir / f'{self.scene_id}.aggregation.json'

        for scene_file_path in [self.ply_path, self.seg_json_path, self.agg_json_path]:
            if not scene_file_path.exists():
                raise FileNotFoundError(f'Missing scene file: {scene_file_path}')

    def _load_point_cloud(self) -> np.ndarray:
        """Load point cloud data from PLY file"""
        vertices = PlyData.read(str(self.ply_path))['vertex']
        return np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    def _init_instance(self) -> List[SceneInstanceMetric]:
        """Convert point cloud into instances"""
        points = self._load_point_cloud()

        with self.seg_json_path.open() as f:
            seg_indices = np.array(json.load(f)['segIndices'])

        with self.agg_json_path.open() as f:
            agg_data = json.load(f)

        instances = []
        for group in agg_data['segGroups']:
            idx_mask = np.isin(seg_indices, group['segments'])
            if np.any(idx_mask):
                instance = SceneInstanceMetric(
                    object_id=group['objectId'],
                    label=group['label'],
                    points=points[idx_mask]
                )
                instance.calc_metrics()
                instances.append(instance)

        return instances

    @staticmethod
    def _calc_pairwise_distances(instances: List[SceneInstanceMetric]) -> dict[str, float]:
        """Calculate the pairwise distances between the object centers"""
        distance_dict = {}
        for idx, inst1 in enumerate(instances):
            for inst2 in instances[idx + 1:]:
                try:
                    # the nearest distance between the surface of two objects
                    hull1 = ConvexHull(inst1.points)
                    hull2 = ConvexHull(inst2.points)
                    distance_dict[f'{inst1.object_id}-{inst2.object_id}'] = float(
                        np.min(distance.cdist(
                            inst1.points[hull1.vertices],
                            inst2.points[hull2.vertices],
                            'euclidean'
                        )))
                except Exception as e:
                    print(f'Error calculating distance between {inst1.object_id} and {inst2.object_id}: {e}')
                    continue

        return distance_dict

    def analyze(self) -> dict[str, any]:
        """Analyze the ScanNet scene and return the scene statistics"""
        instances = self._init_instance()
        pairwise_distance_dict = self._calc_pairwise_distances(instances)

        scene_stats = {
            'scene_id': self.scene_id,
            'num_instances': len(instances),
            # instance details
            'instances': [
                {
                    'object_id': instance.object_id,
                    'label': instance.label,
                    'center': instance.obj_center,
                    'bbox_xyz_min': instance.bbox_xyz_min,
                    'bbox_xyz_max': instance.bbox_xyz_max,
                    'bbox_xyz_len': instance.bbox_xyz_len,
                    'bbox_volume': instance.bbox_volume
                }
                for instance in instances
            ],
            # pairwise distances between the surfaces of the instances
            'pairwise_distances': dict(sorted(
                pairwise_distance_dict.items(),
                key=lambda item: tuple(map(int, item[0].split('-')))
            ))
        }

        return scene_stats


def process_scene(scene_dir: str, export_dir: str, export_prefix: str) -> None:
    try:
        analyzer = ScanNetSceneAnalyzer(scene_dir)
        scene_stats = analyzer.analyze()

        export_dict_as_json_file(
            scene_stats,
            os.path.join(export_dir, f'{export_prefix}-{analyzer.scene_id}.json')
        )
    except Exception as e:
        print(f'Error processing scene {scene_dir}: {e}')


@click.command()
@click.option('--scenes',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
              prompt='Enter the ScanNet scene directory path',
              help='The directory containing the ScanNet scene folders')
@click.option('--export_dir', default='./output/scene_stats/',
              type=click.Path(file_okay=False, writable=True),
              help='The directory to export the scene statistics JSON files')
@click.option('--export_prefix', default='scene_stats', type=str,
              help='The prefix for the exported scene statistics JSON files. Default is "scene_stats"'
                   '(e.g., scene_001 -> scene_stats-scene_001.json)')
@click.option('--n_jobs', default=-1, type=click.IntRange(-1, None),
              help='Number of parallel jobs to process the scenes')
@click.option('-s', '--skip_confirm', is_flag=True,
              help='Skip the confirmation prompt before processing the scenes')
def cli(scenes, export_dir, export_prefix, n_jobs, skip_confirm):
    subfolders = [f.path for f in os.scandir(scenes) if f.is_dir()]
    scene_folders = subfolders if len(subfolders) > 0 else [scenes]

    print(f'Found {len(scene_folders)} to be processed with '
          f'{n_jobs if n_jobs != -1 else os.cpu_count()} process(es):\n')
    print('\n'.join(scene_folders if len(scene_folders) <= 6 else
                    scene_folders[:3] + ['...'] + scene_folders[-3:]))
    print(f'Summarized JSON files will be exported to {os.path.abspath(export_dir)}.')
    if not skip_confirm and not click.confirm('Proceed?', default=True):
        return
    os.makedirs(export_dir, exist_ok=True)
    # if export path is not empty, ask whether remove the existing files
    if os.listdir(export_dir):
        if not click.confirm(f'Files already exist in {export_dir}. Remove existing files?', default=True):
            return
        for file in os.listdir(export_dir):
            os.remove(os.path.join(export_dir, file))

    ParallelTqdm(n_jobs=n_jobs)(
        [delayed(process_scene)(scene_dir, export_dir, export_prefix)
         for scene_dir in scene_folders]
    )


if __name__ == '__main__':
    cli()
