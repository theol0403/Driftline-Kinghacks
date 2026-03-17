from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from .types import Detection


class ArtifactStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        path = self.base_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def preview_relpath(self, job_id: str) -> str:
        return f"{job_id}/preview/latest.jpg"

    def preview_abspath(self, job_id: str) -> Path:
        path = self._job_dir(job_id) / "preview"
        path.mkdir(parents=True, exist_ok=True)
        return path / "latest.jpg"

    def crop_relpath(self, job_id: str, frame_index: int, detection_index: int) -> str:
        return f"{job_id}/crops/frame_{frame_index:06d}_{detection_index:03d}.jpg"

    def crop_abspath(self, job_id: str, frame_index: int, detection_index: int) -> Path:
        path = self._job_dir(job_id) / "crops"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"frame_{frame_index:06d}_{detection_index:03d}.jpg"

    def save_preview_frame(
        self,
        job_id: str,
        frame_bgr,
        detections: Iterable[Detection],
    ) -> str:
        annotated = annotate_frame(frame_bgr, detections)
        destination = self.preview_abspath(job_id)
        cv2.imwrite(str(destination), annotated)
        return self.preview_relpath(job_id)

    def save_detection_crop(
        self,
        job_id: str,
        frame_index: int,
        detection_index: int,
        frame_bgr,
        detection: Detection,
    ) -> str | None:
        xmin, ymin, xmax, ymax = detection.bbox
        height, width = frame_bgr.shape[:2]
        x0 = max(0, min(width, int(xmin)))
        x1 = max(0, min(width, int(xmax)))
        y0 = max(0, min(height, int(ymin)))
        y1 = max(0, min(height, int(ymax)))
        if x1 <= x0 or y1 <= y0:
            return None

        crop = frame_bgr[y0:y1, x0:x1]
        if max(crop.shape[:2]) > 256:
            scale = 256.0 / max(crop.shape[:2])
            crop = cv2.resize(
                crop,
                (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            )

        destination = self.crop_abspath(job_id, frame_index, detection_index)
        cv2.imwrite(str(destination), crop)
        return self.crop_relpath(job_id, frame_index, detection_index)


def color_for_category(category: str | None) -> tuple[int, int, int]:
    key = (category or "").lower()
    if "pothole" in key:
        return (91, 73, 209)
    if "crack" in key:
        return (237, 111, 47)
    return (128, 128, 128)


def annotate_frame(frame_bgr, detections: Iterable[Detection]):
    annotated = frame_bgr.copy()
    for detection in detections:
        xmin, ymin, xmax, ymax = [int(value) for value in detection.bbox]
        category = detection.category or detection.label
        color = color_for_category(category)
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{category} {detection.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (xmin, max(18, ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
