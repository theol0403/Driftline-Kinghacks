from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox: Tuple[float, float, float, float]
    category: str | None = None


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ProjectedDetection:
    job_id: str
    frame_index: int
    frame_ts: float
    category: str
    label: str
    score: float
    vehicle_lat: float
    vehicle_lon: float
    vehicle_heading_deg: float
    hazard_lat: float
    hazard_lon: float
    offset_forward_m: float
    offset_lateral_m: float
    local_x_m: float
    local_y_m: float
    crop_path: Optional[str] = None
    observed_at_iso: Optional[str] = None


@dataclass(frozen=True)
class HazardCluster:
    job_id: Optional[str]
    category: str
    centroid_lat: float
    centroid_lon: float
    detection_count: int
    unique_frame_count: int
    avg_confidence: float
    first_seen: float
    last_seen: float
    severity_score: float
    verified: bool
    observed_at_iso: str
    thumbnail_path: Optional[str] = None


@dataclass(frozen=True)
class ProgressSnapshot:
    job_id: str
    status: str
    frames_processed: int
    total_frames: int
    raw_detection_count: int
    verified_hazard_count: int = 0
    error: Optional[str] = None
    preview_image_path: Optional[str] = None


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: str
    frames_processed: int
    total_frames: int
    raw_detection_count: int
    verified_hazard_count: int
    distance_km: float
    credits_earned: float
