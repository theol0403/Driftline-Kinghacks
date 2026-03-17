from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Callable, Optional

from .artifacts import ArtifactStore
from .clustering import cluster_projected_detections, compute_severity_score, parse_iso_datetime
from .detector import DetectorConfig, UltralyticsDetector, filter_by_category
from .gps import GpsTrack, load_gps_csv
from .mapping import MappingConfig, OccupancyGridMapper, project_detection_to_geo
from .storage import DriftlineRepository
from .types import Detection, HazardCluster, JobResult, Pose2D, ProgressSnapshot, ProjectedDetection
from .vo import VisualOdometry, VisualOdometryConfig


ProgressCallback = Callable[[ProgressSnapshot], None]


@dataclass
class PipelineConfig:
    source: str
    gps_csv: str
    database_path: str
    job_id: str
    model: str
    artifacts_dir: str
    conf: float = 0.25
    imgsz: int = 640
    categories: Optional[list[str]] = None
    no_vo: bool = False
    vo_scale: float = 0.02
    map_width: float = 50.0
    map_height: float = 50.0
    map_resolution: float = 0.2
    label_map_path: Optional[str] = None
    rerun_recording: Optional[str] = None
    enable_rerun: bool = False
    progress_every: int = 10


def load_label_map(path: Optional[str]) -> Optional[dict[str, str]]:
    if not path:
        return None
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def open_capture(source: str):
    import cv2

    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import atan2, cos, radians, sin, sqrt

    radius_m = 6_371_000.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    a = sin(d_lat / 2.0) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(d_lon / 2.0) ** 2
    return 2.0 * radius_m * atan2(sqrt(a), sqrt(1.0 - a))


def route_distance_km(track: GpsTrack) -> float:
    samples = track.samples
    if len(samples) < 2:
        return 0.0
    total_m = 0.0
    for before, after in zip(samples, samples[1:]):
        total_m += haversine_distance_m(before.lat, before.lon, after.lat, after.lon)
    return total_m / 1000.0


def compute_credits(distance_km: float, verified_hazard_count: int) -> float:
    return round((0.05 * distance_km) + (0.25 * verified_hazard_count), 2)


def _emit_progress(
    repository: DriftlineRepository,
    snapshot: ProgressSnapshot,
    callback: ProgressCallback | None,
) -> None:
    repository.update_job_progress(
        snapshot.job_id,
        status=snapshot.status,
        frames_processed=snapshot.frames_processed,
        total_frames=snapshot.total_frames,
        raw_detection_count=snapshot.raw_detection_count,
        verified_hazard_count=snapshot.verified_hazard_count,
        error=snapshot.error,
        latest_preview_path=snapshot.preview_image_path,
    )
    if callback is not None:
        callback(snapshot)


def _finalize_clusters(
    clusters: list[HazardCluster],
    completed_at: str,
) -> list[HazardCluster]:
    completion_dt = parse_iso_datetime(completed_at)
    return [
        replace(
            cluster,
            observed_at_iso=completed_at,
            severity_score=compute_severity_score(
                detection_count=cluster.detection_count,
                avg_confidence=cluster.avg_confidence,
                observed_at_iso=completed_at,
                now=completion_dt,
            ),
        )
        for cluster in clusters
    ]


def run_pipeline(
    config: PipelineConfig,
    progress_callback: ProgressCallback | None = None,
) -> JobResult:
    import cv2

    repository = DriftlineRepository(config.database_path)
    repository.initialize()
    artifact_store = ArtifactStore(config.artifacts_dir)

    if not os.path.exists(config.gps_csv):
        raise FileNotFoundError(f"GPS CSV not found at {config.gps_csv}")

    gps_track = load_gps_csv(config.gps_csv)
    if not gps_track.samples:
        raise ValueError("GPS CSV did not contain any samples")

    cap = open_capture(config.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {config.source}")

    label_map = load_label_map(config.label_map_path)
    detector = UltralyticsDetector(
        DetectorConfig(
            model_path=config.model,
            conf=config.conf,
            imgsz=config.imgsz,
            label_map=label_map,
        )
    )
    mapper = OccupancyGridMapper(
        MappingConfig(
            width_m=config.map_width,
            height_m=config.map_height,
            resolution_m=config.map_resolution,
        )
    )

    vo = None
    pose = Pose2D(0.0, 0.0, 0.0)
    if not config.no_vo:
        vo = VisualOdometry(VisualOdometryConfig(scale_m_per_px=config.vo_scale))

    rerun_logger = None
    if config.enable_rerun:
        from .rerun_viz import RerunConfig, RerunLogger

        rerun_logger = RerunLogger(
            RerunConfig(recording_path=config.rerun_recording, spawn=True)
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    origin = gps_track.samples[0]
    observed_at_iso = datetime.now(UTC).isoformat()
    raw_detections: list[ProjectedDetection] = []
    raw_detection_count = 0
    frame_index = 0
    latest_preview_path: str | None = None

    _emit_progress(
        repository,
        ProgressSnapshot(
            job_id=config.job_id,
            status="processing",
            frames_processed=0,
            total_frames=total_frames,
            raw_detection_count=0,
            preview_image_path=None,
        ),
        progress_callback,
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1
            t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t_s <= 0.0 and fps > 0.0:
                t_s = frame_index / fps

            detections: list[Detection] = detector.detect(frame)
            detections = filter_by_category(detections, config.categories)

            if vo is not None:
                pose_update = vo.update(frame)
                if pose_update is not None:
                    pose = pose_update

            points = mapper.update(pose, detections, frame.shape)
            gps_sample = gps_track.nearest(t_s)
            frame_projected: list[ProjectedDetection] = []
            if gps_sample is not None:
                for detection_index, detection in enumerate(detections):
                    crop_path = artifact_store.save_detection_crop(
                        config.job_id,
                        frame_index,
                        detection_index,
                        frame,
                        detection,
                    )
                    frame_projected.append(
                        project_detection_to_geo(
                            job_id=config.job_id,
                            frame_index=frame_index,
                            frame_ts=t_s,
                            detection=detection,
                            frame_shape=frame.shape,
                            vehicle_lat=gps_sample.lat,
                            vehicle_lon=gps_sample.lon,
                            vehicle_heading_deg=gps_sample.heading_deg,
                            origin_lat=origin.lat,
                            origin_lon=origin.lon,
                            config=mapper.config,
                            crop_path=crop_path,
                            observed_at_iso=observed_at_iso,
                        )
                    )

            if frame_projected:
                repository.insert_raw_detections(frame_projected)
                raw_detections.extend(frame_projected)
                raw_detection_count += len(frame_projected)

            should_publish_preview = (
                frame_index == 1 or frame_index % max(config.progress_every, 1) == 0
            )
            if should_publish_preview:
                latest_preview_path = artifact_store.save_preview_frame(
                    config.job_id,
                    frame,
                    detections,
                )

            if rerun_logger is not None:
                import rerun as rr

                rr.set_time("time", duration=t_s)
                rerun_logger.log_frame(frame)
                rerun_logger.log_detections(detections)
                rerun_logger.log_pose(pose)
                if gps_sample is not None:
                    rerun_logger.log_gps(gps_sample.lat, gps_sample.lon)
                    rerun_logger.log_projected_hazards(frame_projected)
                rerun_logger.log_detection_points(points)
                rerun_logger.log_grid(mapper.grid_image())

            if should_publish_preview:
                _emit_progress(
                    repository,
                    ProgressSnapshot(
                        job_id=config.job_id,
                        status="processing",
                        frames_processed=frame_index,
                        total_frames=total_frames,
                        raw_detection_count=raw_detection_count,
                        preview_image_path=latest_preview_path,
                    ),
                    progress_callback,
                )

        if latest_preview_path is None and frame_index > 0:
            latest_preview_path = artifact_store.preview_relpath(config.job_id)

        provisional_clusters = cluster_projected_detections(raw_detections)
        verified_count = sum(1 for cluster in provisional_clusters if cluster.verified)
        distance_km = route_distance_km(gps_track)
        credits_earned = compute_credits(distance_km, verified_count)
        completed_at = repository.mark_job_completed(
            config.job_id,
            frames_processed=frame_index,
            total_frames=total_frames,
            raw_detection_count=raw_detection_count,
            verified_hazard_count=verified_count,
            distance_km=distance_km,
            credits_earned=credits_earned,
            latest_preview_path=latest_preview_path,
        )
        repository.replace_job_clusters(
            config.job_id,
            _finalize_clusters(provisional_clusters, completed_at),
        )

        _emit_progress(
            repository,
            ProgressSnapshot(
                job_id=config.job_id,
                status="completed",
                frames_processed=frame_index,
                total_frames=total_frames,
                raw_detection_count=raw_detection_count,
                verified_hazard_count=verified_count,
                preview_image_path=latest_preview_path,
            ),
            progress_callback,
        )

        return JobResult(
            job_id=config.job_id,
            status="completed",
            frames_processed=frame_index,
            total_frames=total_frames,
            raw_detection_count=raw_detection_count,
            verified_hazard_count=verified_count,
            distance_km=distance_km,
            credits_earned=credits_earned,
        )
    except Exception as exc:
        repository.mark_job_error(config.job_id, str(exc))
        _emit_progress(
            repository,
            ProgressSnapshot(
                job_id=config.job_id,
                status="error",
                frames_processed=frame_index,
                total_frames=total_frames,
                raw_detection_count=raw_detection_count,
                error=str(exc),
                preview_image_path=latest_preview_path,
            ),
            progress_callback,
        )
        raise
    finally:
        cap.release()
