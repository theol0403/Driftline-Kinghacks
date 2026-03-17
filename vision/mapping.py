from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .types import Detection, Pose2D, ProjectedDetection


EARTH_RADIUS_M = 6_371_000.0


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
        points: List[Tuple[float, float, str]] = []
        for detection in detections:
            distance_m, lateral_m = estimate_detection_offset(
                detection,
                frame_shape,
                self.config,
            )

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


def estimate_detection_offset(
    detection: Detection,
    frame_shape: Sequence[int],
    config: MappingConfig,
) -> Tuple[float, float]:
    height, width = frame_shape[:2]
    del height
    if width <= 0:
        return config.min_distance_m, 0.0

    xmin, ymin, xmax, ymax = detection.bbox
    bbox_h = max(1.0, ymax - ymin)
    distance_m = config.distance_scale / bbox_h
    distance_m = float(
        np.clip(
            distance_m,
            config.min_distance_m,
            config.max_distance_m,
        )
    )
    x_center = (xmin + xmax) / 2.0
    lateral_m = ((x_center / width) - 0.5) * distance_m * config.lateral_scale
    return distance_m, lateral_m


def heading_to_east_north(
    forward_m: float,
    lateral_m: float,
    heading_deg: float,
) -> Tuple[float, float]:
    heading_rad = math.radians(heading_deg)
    east_m = forward_m * math.sin(heading_rad) + lateral_m * math.cos(heading_rad)
    north_m = forward_m * math.cos(heading_rad) - lateral_m * math.sin(heading_rad)
    return east_m, north_m


def offset_to_latlon(
    lat: float,
    lon: float,
    east_m: float,
    north_m: float,
) -> Tuple[float, float]:
    delta_lat = math.degrees(north_m / EARTH_RADIUS_M)
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    delta_lon = math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
    return lat + delta_lat, lon + delta_lon


def latlon_to_local_xy(
    lat: float,
    lon: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    mean_lat = math.radians((lat + origin_lat) / 2.0)
    x_m = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(mean_lat)
    y_m = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x_m, y_m


def project_detection_to_geo(
    *,
    job_id: str,
    frame_index: int,
    frame_ts: float,
    detection: Detection,
    frame_shape: Sequence[int],
    vehicle_lat: float,
    vehicle_lon: float,
    vehicle_heading_deg: float,
    origin_lat: float,
    origin_lon: float,
    config: MappingConfig,
    crop_path: str | None = None,
    observed_at_iso: str | None = None,
) -> ProjectedDetection:
    category = detection.category or detection.label
    forward_m, lateral_m = estimate_detection_offset(detection, frame_shape, config)
    east_m, north_m = heading_to_east_north(forward_m, lateral_m, vehicle_heading_deg)
    hazard_lat, hazard_lon = offset_to_latlon(vehicle_lat, vehicle_lon, east_m, north_m)
    local_x_m, local_y_m = latlon_to_local_xy(hazard_lat, hazard_lon, origin_lat, origin_lon)
    return ProjectedDetection(
        job_id=job_id,
        frame_index=frame_index,
        frame_ts=frame_ts,
        category=category,
        label=detection.label,
        score=detection.score,
        vehicle_lat=vehicle_lat,
        vehicle_lon=vehicle_lon,
        vehicle_heading_deg=vehicle_heading_deg,
        hazard_lat=hazard_lat,
        hazard_lon=hazard_lon,
        offset_forward_m=forward_m,
        offset_lateral_m=lateral_m,
        local_x_m=local_x_m,
        local_y_m=local_y_m,
        crop_path=crop_path,
        observed_at_iso=observed_at_iso,
    )
