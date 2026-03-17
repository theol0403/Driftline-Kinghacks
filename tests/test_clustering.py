from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vision.clustering import cluster_projected_detections, compute_severity_score
from vision.mapping import MappingConfig, project_detection_to_geo
from vision.types import Detection, ProjectedDetection


def make_projected_detection(
    *,
    job_id: str,
    frame_index: int,
    lat: float,
    lon: float,
    category: str,
    score: float,
    observed_at_iso: str,
) -> ProjectedDetection:
    return ProjectedDetection(
        job_id=job_id,
        frame_index=frame_index,
        frame_ts=float(frame_index),
        category=category,
        label=category,
        score=score,
        vehicle_lat=lat,
        vehicle_lon=lon,
        vehicle_heading_deg=0.0,
        hazard_lat=lat,
        hazard_lon=lon,
        offset_forward_m=0.0,
        offset_lateral_m=0.0,
        local_x_m=0.0,
        local_y_m=0.0,
        crop_path=f"{job_id}/crops/frame_{frame_index:06d}.jpg",
        observed_at_iso=observed_at_iso,
    )


def test_project_detection_to_geo_moves_forward_and_right_with_north_heading() -> None:
    detection = Detection(
        label="D40",
        score=0.92,
        bbox=(120.0, 40.0, 160.0, 140.0),
        category="pothole",
    )

    projected = project_detection_to_geo(
        job_id="job-1",
        frame_index=1,
        frame_ts=1.0,
        detection=detection,
        frame_shape=(200, 200, 3),
        vehicle_lat=44.2312,
        vehicle_lon=-76.486,
        vehicle_heading_deg=0.0,
        origin_lat=44.2312,
        origin_lon=-76.486,
        config=MappingConfig(),
    )

    assert projected.hazard_lat > projected.vehicle_lat
    assert projected.hazard_lon > projected.vehicle_lon
    assert projected.local_x_m > 0.0
    assert projected.local_y_m > 0.0


def test_dbscan_verifies_three_same_category_frames_only() -> None:
    observed_at_iso = datetime.now(UTC).isoformat()
    detections = [
        make_projected_detection(
            job_id="job-1",
            frame_index=1,
            lat=44.231200,
            lon=-76.486000,
            category="pothole",
            score=0.91,
            observed_at_iso=observed_at_iso,
        ),
        make_projected_detection(
            job_id="job-1",
            frame_index=2,
            lat=44.231206,
            lon=-76.486002,
            category="pothole",
            score=0.89,
            observed_at_iso=observed_at_iso,
        ),
        make_projected_detection(
            job_id="job-1",
            frame_index=3,
            lat=44.231212,
            lon=-76.486004,
            category="pothole",
            score=0.93,
            observed_at_iso=observed_at_iso,
        ),
        make_projected_detection(
            job_id="job-1",
            frame_index=4,
            lat=44.231202,
            lon=-76.486001,
            category="longitudinal_crack",
            score=0.77,
            observed_at_iso=observed_at_iso,
        ),
    ]

    clusters = cluster_projected_detections(detections)

    pothole_clusters = [cluster for cluster in clusters if cluster.category == "pothole"]
    crack_clusters = [
        cluster for cluster in clusters if cluster.category == "longitudinal_crack"
    ]

    assert len(pothole_clusters) == 1
    assert pothole_clusters[0].verified is True
    assert pothole_clusters[0].detection_count == 3
    assert pothole_clusters[0].unique_frame_count == 3
    assert pothole_clusters[0].thumbnail_path is not None

    assert len(crack_clusters) == 1
    assert crack_clusters[0].verified is False
    assert crack_clusters[0].detection_count == 1


def test_severity_score_favors_recent_clusters() -> None:
    now = datetime.now(UTC)
    recent = compute_severity_score(
        detection_count=3,
        avg_confidence=0.8,
        observed_at_iso=now.isoformat(),
        now=now,
    )
    stale = compute_severity_score(
        detection_count=3,
        avg_confidence=0.8,
        observed_at_iso=(now - timedelta(days=21)).isoformat(),
        now=now,
    )

    assert recent > stale
