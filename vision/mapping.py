from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .types import Detection, Pose2D


@dataclass
class MappingConfig:
    width_m: float = 50.0
    height_m: float = 50.0
    resolution_m: float = 0.2
    distance_scale: float = 1200.0
    min_distance_m: float = 2.0
    max_distance_m: float = 25.0
    lateral_scale: float = 1.6


class OccupancyGridMapper:
    def __init__(self, config: MappingConfig) -> None:
        self.config = config
        self.grid_height = int(config.height_m / config.resolution_m)
        self.grid_width = int(config.width_m / config.resolution_m)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.origin_row = self.grid_height // 2
        self.origin_col = self.grid_width // 2

    def _world_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        row = int(self.origin_row - x_m / self.config.resolution_m)
        col = int(self.origin_col + y_m / self.config.resolution_m)
        return row, col

    def update(
        self, pose: Pose2D, detections: Iterable[Detection], frame_shape: Tuple[int, int]
    ) -> List[Tuple[float, float, str]]:
        height, width = frame_shape[:2]
        points: List[Tuple[float, float, str]] = []
        if width > 0:
            for detection in detections:
                xmin, ymin, xmax, ymax = detection.bbox
                bbox_h = max(1.0, ymax - ymin)
                distance_m = self.config.distance_scale / bbox_h
                distance_m = float(
                    np.clip(
                        distance_m,
                        self.config.min_distance_m,
                        self.config.max_distance_m,
                    )
                )
                x_center = (xmin + xmax) / 2.0
                lateral_m = (
                    (x_center / width) - 0.5
                ) * distance_m * self.config.lateral_scale

                cos_yaw = math.cos(pose.yaw)
                sin_yaw = math.sin(pose.yaw)
                world_x = pose.x + distance_m * cos_yaw - lateral_m * sin_yaw
                world_y = pose.y + distance_m * sin_yaw + lateral_m * cos_yaw
                points.append((world_x, world_y, detection.category or detection.label))

                row, col = self._world_to_grid(world_x, world_y)
                if 0 <= row < self.grid_height and 0 <= col < self.grid_width:
                    self.grid[row, col] = min(255.0, self.grid[row, col] + 12.0)
        return points

    def grid_image(self) -> np.ndarray:
        return self.grid.astype(np.uint8)
