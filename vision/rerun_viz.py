from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .types import Detection, Pose2D


@dataclass
class RerunConfig:
    app_id: str = "kingston-vision"
    spawn: bool = True
    recording_path: Optional[str] = None
    map_origin: str = "map"
    map_zoom: float = 16.0
    map_provider: rrb.MapProvider = rrb.MapProvider.OpenStreetMap


@dataclass
class RerunLogger:
    config: RerunConfig
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    geo_path: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        rr.init(self.config.app_id, spawn=self.config.spawn)
        if self.config.recording_path:
            rr.save(self.config.recording_path)
        self._setup_blueprint()

    def _setup_blueprint(self) -> None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.MapView(
                    origin=self.config.map_origin,
                    name="Kingston Map",
                    zoom=self.config.map_zoom,
                    background=self.config.map_provider,
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="camera", name="Camera"),
                    rrb.Spatial2DView(origin="camera/detections", name="Detections"),
                ),
                column_shares=[0.45, 0.55],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

    def log_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = frame_bgr[:, :, ::-1]
        rr.log("camera", rr.Image(frame_rgb))

    def log_detections(self, detections: Iterable[Detection]) -> None:
        dets = list(detections)
        if not dets:
            return
        boxes = [det.bbox for det in dets]
        labels = [det.category or det.label for det in dets]
        rr.log(
            "camera/detections",
            rr.Boxes2D(
                array=boxes,
                array_format=rr.Box2DFormat.XYXY,
                labels=labels,
                class_ids=None,
            ),
        )

    def log_pose(self, pose: Pose2D) -> None:
        self.trajectory.append((pose.x, pose.y))
        rr.log(
            "world/vehicle",
            rr.Transform3D(
                translation=[pose.x, pose.y, 0.0],
                rotation=rr.RotationAxisAngle([0.0, 0.0, 1.0], pose.yaw),
            ),
        )
        if len(self.trajectory) > 1:
            points = [[x, y, 0.0] for x, y in self.trajectory]
            rr.log("world/trajectory", rr.LineStrips3D(points))

    def log_gps(self, lat: float, lon: float) -> None:
        self.geo_path.append((lat, lon))
        rr.log(
            f"{self.config.map_origin}/vehicle",
            rr.GeoPoints(lat_lon=[[lat, lon]]),
        )
        if len(self.geo_path) > 1:
            rr.log(
                f"{self.config.map_origin}/path",
                rr.GeoLineStrings(lat_lon=[self.geo_path]),
            )

    def log_detection_points(self, points: Sequence[Tuple[float, float, str]]) -> None:
        if not points:
            return
        positions = [[x, y, 0.0] for x, y, _ in points]
        labels = [label for _, _, label in points]
        rr.log("world/detection_points", rr.Points3D(positions, labels=labels))

    def log_grid(self, grid: np.ndarray) -> None:
        rr.log("map/occupancy", rr.Image(grid))
