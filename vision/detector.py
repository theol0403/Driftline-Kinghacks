from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .types import Detection


RDD2022_POTHOLE_LABELS = {"D00", "D10", "D20", "D40"}


def default_label_map() -> Dict[str, str]:
    return {
        "D00": "longitudinal_crack",
        "D10": "transverse_crack",
        "D20": "alligator_crack",
        "D40": "pothole",
    }


def map_label(label: str, label_map: Dict[str, str]) -> str:
    if label in label_map:
        return label_map[label]
    lower = label.lower()
    if "pothole" in lower:
        return "potholes"
    if "ice" in lower:
        return "ice_patches"
    if "sidewalk" in lower or "snow" in lower or "blocked" in lower:
        return "blocked_sidewalks"
    if "trash" in lower or "garbage" in lower:
        return "garbage_left_out"
    if "person" in lower or "pedestrian" in lower:
        return "foot_traffic"
    return label


@dataclass
class DetectorConfig:
    model_path: str
    conf: float = 0.25
    imgsz: int = 640
    label_map: Optional[Dict[str, str]] = None


class UltralyticsDetector:
    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO

        self.model = YOLO(config.model_path)
        self.conf = config.conf
        self.imgsz = config.imgsz
        self.label_map = config.label_map or default_label_map()

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        detections: List[Detection] = []
        names = result.names or {}
        for box in result.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id))
            category = map_label(label, self.label_map)
            detections.append(
                Detection(
                    label=label,
                    score=score,
                    bbox=(xmin, ymin, xmax, ymax),
                    category=category,
                )
            )
        return detections


def filter_by_category(
    detections: Iterable[Detection], categories: Optional[Iterable[str]]
) -> List[Detection]:
    if not categories:
        return list(detections)
    allowed = {category.lower() for category in categories}
    return [det for det in detections if det.category and det.category.lower() in allowed]
