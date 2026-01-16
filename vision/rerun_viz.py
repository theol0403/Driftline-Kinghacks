from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
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
    detection_index: int = 0

    def __post_init__(self) -> None:
        rr.init(self.config.app_id, spawn=self.config.spawn)
        if self.config.recording_path:
            rr.save(self.config.recording_path)
        self._setup_blueprint()

    def _setup_blueprint(self) -> None:
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.MapView(
                        origin=self.config.map_origin,
                        name="Kingston Map",
                        zoom=self.config.map_zoom,
                        background=self.config.map_provider,
                    ),
                    rrb.Spatial2DView(origin="camera", name="Camera"),
                    column_shares=[0.45, 0.55],
                ),
                rrb.TimeSeriesView(origin="metrics/detections", name="Detections Timeline"),
                row_shares=[0.75, 0.25],
            ),
            rrb.SelectionPanel(expanded=True),
            rrb.TimePanel(expanded=True),
            collapse_panels=False,
        )
        rr.send_blueprint(blueprint)

    def _color_for_label(self, label: str) -> List[int]:
        key = label.lower()
        if "pothole" in key:
            return [220, 50, 50]
        if "crack" in key:
            return [150, 200, 255]
        return [180, 180, 180]

    def log_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = frame_bgr[:, :, ::-1]
        rr.log("camera", rr.Image(frame_rgb))

    def log_detections(self, detections: Iterable[Detection]) -> None:
        dets = list(detections)
        labels = [det.category or det.label for det in dets]
        label_counts = Counter(labels)
        rr.log("metrics/detections/total", rr.Scalars([len(dets)]))
        for label, count in label_counts.items():
            rr.log(f"metrics/detections/by_label/{label}", rr.Scalars([count]))
        if not dets:
            return
        boxes = [det.bbox for det in dets]
        colors = [self._color_for_label(label) for label in labels]
        rr.log(
            "camera/detections",
            rr.Boxes2D(
                array=boxes,
                array_format=rr.Box2DFormat.XYXY,
                labels=labels,
                colors=colors,
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

    def log_gps_detections(
        self,
        lat: float,
        lon: float,
        detections: Iterable[Detection],
        frame_bgr: np.ndarray,
    ) -> None:
        height, width = frame_bgr.shape[:2]
        for det in detections:
            label = det.category or det.label
            color = self._color_for_label(label)
            radius = (2.5 if "pothole" in label.lower() else 1.2) + 1.6 * det.score
            xmin, ymin, xmax, ymax = det.bbox
            x0 = max(0, min(width, int(xmin)))
            x1 = max(0, min(width, int(xmax)))
            y0 = max(0, min(height, int(ymin)))
            y1 = max(0, min(height, int(ymax)))
            if x1 <= x0 or y1 <= y0:
                continue
            crop_bgr = frame_bgr[y0:y1, x0:x1]
            if max(crop_bgr.shape[:2]) > 192:
                scale = 192.0 / max(crop_bgr.shape[:2])
                new_w = max(1, int(crop_bgr.shape[1] * scale))
                new_h = max(1, int(crop_bgr.shape[0] * scale))
                crop_bgr = cv2.resize(crop_bgr, (new_w, new_h))
            crop_rgb = crop_bgr[:, :, ::-1]
            entity = f"{self.config.map_origin}/detections/items/{self.detection_index:06d}"
            self.detection_index += 1
            rr.log(
                entity,
                rr.GeoPoints(
                    lat_lon=[[lat, lon]],
                    radii=rr.Radius.ui_points(radius),
                    colors=[color],
                ),
            )
            rr.log(entity, rr.TextLog(f"{det.label} ({det.score:.2f})"))
            rr.log(entity, rr.Image(crop_rgb))

    def log_detection_points(self, points: Sequence[Tuple[float, float, str]]) -> None:
        if not points:
            return
        positions = [[x, y, 0.0] for x, y, _ in points]
        labels = [label for _, _, label in points]
        rr.log("world/detection_points", rr.Points3D(positions, labels=labels))

    def log_grid(self, grid: np.ndarray) -> None:
        rr.log("map/occupancy", rr.Image(grid))
