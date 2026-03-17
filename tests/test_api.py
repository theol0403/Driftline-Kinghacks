from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from vision.clustering import cluster_projected_detections
from vision.pipeline import compute_credits
from vision.storage import DriftlineRepository
from vision.types import ProjectedDetection
from webapp.app import create_app


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


def seed_completed_job(
    repository: DriftlineRepository,
    *,
    job_id: str,
    video_filename: str,
    detections: list[ProjectedDetection],
    distance_km: float,
    completed_at: str,
    preview_path: str | None = None,
) -> None:
    repository.create_job(
        job_id=job_id,
        video_filename=video_filename,
        gps_filename=f"{job_id}.csv",
    )
    repository.insert_raw_detections(detections)
    clusters = cluster_projected_detections(detections)
    verified_count = sum(1 for cluster in clusters if cluster.verified)
    repository.mark_job_completed(
        job_id,
        frames_processed=len(detections),
        total_frames=len(detections),
        raw_detection_count=len(detections),
        verified_hazard_count=verified_count,
        distance_km=distance_km,
        credits_earned=compute_credits(distance_km, verified_count),
        latest_preview_path=preview_path,
        completed_at=completed_at,
    )
    repository.replace_job_clusters(job_id, clusters)


def create_test_client(tmp_path: Path) -> TestClient:
    static_dir = Path(__file__).resolve().parents[1] / "webapp" / "static"
    app = create_app(
        database_path=tmp_path / "driftline.sqlite",
        upload_dir=tmp_path / "uploads",
        static_dir=static_dir,
    )
    return TestClient(app)


def test_upload_requires_gps_csv(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.post(
        "/api/jobs",
        files={"video": ("route.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "gps_csv is required"


def test_job_results_and_dashboard_priority_order(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    repository = client.app.state.repository
    now = datetime.now(UTC)

    seed_completed_job(
        repository,
        job_id="job-high",
        video_filename="high.mp4",
        distance_km=2.0,
        completed_at=now.isoformat(),
        preview_path="job-high/preview/latest.jpg",
        detections=[
            make_projected_detection(
                job_id="job-high",
                frame_index=index,
                lat=44.23120 + index * 0.000005,
                lon=-76.48600 + index * 0.000003,
                category="pothole",
                score=0.95,
                observed_at_iso=now.isoformat(),
            )
            for index in range(1, 4)
        ],
    )
    seed_completed_job(
        repository,
        job_id="job-low",
        video_filename="low.mp4",
        distance_km=1.0,
        completed_at=(now - timedelta(days=10)).isoformat(),
        preview_path="job-low/preview/latest.jpg",
        detections=[
            make_projected_detection(
                job_id="job-low",
                frame_index=index,
                lat=44.23160 + index * 0.000005,
                lon=-76.48630 + index * 0.000003,
                category="pothole",
                score=0.7,
                observed_at_iso=(now - timedelta(days=10)).isoformat(),
            )
            for index in range(1, 4)
        ],
    )

    results_response = client.get("/api/jobs/job-high/results")
    assert results_response.status_code == 200
    results_body = results_response.json()
    assert results_body["job"]["status"] == "completed"
    assert results_body["job"]["preview_image_url"] == "/artifacts/job-high/preview/latest.jpg"
    assert len(results_body["hazards"]) == 1
    assert results_body["rebates"]["verified_hazard_count"] == 1
    assert results_body["hazards"][0]["thumbnail_url"] == "/artifacts/job-high/crops/frame_000001.jpg"

    dashboard_response = client.get("/api/dashboard")
    assert dashboard_response.status_code == 200
    dashboard_body = dashboard_response.json()
    assert dashboard_body["summary"]["total_jobs"] == 2
    assert len(dashboard_body["priority_repairs"]) == 2
    assert dashboard_body["priority_repairs"][0]["thumbnail_url"].startswith("/artifacts/")
    assert (
        dashboard_body["priority_repairs"][0]["severity_score"]
        >= dashboard_body["priority_repairs"][1]["severity_score"]
    )


def test_community_endpoint_merges_cross_job_hazards(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    repository = client.app.state.repository
    observed_at = datetime.now(UTC).isoformat()

    for job_id, lat, lon in (
        ("job-a", 44.23120, -76.48600),
        ("job-b", 44.23121, -76.48599),
    ):
        seed_completed_job(
            repository,
            job_id=job_id,
            video_filename=f"{job_id}.mp4",
            distance_km=1.5,
            completed_at=observed_at,
            preview_path=f"{job_id}/preview/latest.jpg",
            detections=[
                make_projected_detection(
                    job_id=job_id,
                    frame_index=index,
                    lat=lat + index * 0.000002,
                    lon=lon + index * 0.000001,
                    category="pothole",
                    score=0.88,
                    observed_at_iso=observed_at,
                )
                for index in range(1, 4)
            ],
        )

    response = client.get("/api/community")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["completed_jobs"] == 2
    assert len(body["hazards"]) == 1
    assert body["hazards"][0]["category"] == "pothole"
    assert body["hazards"][0]["thumbnail_url"].startswith("/artifacts/")
