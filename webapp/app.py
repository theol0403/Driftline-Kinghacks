from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vision.clustering import cluster_completed_jobs
from vision.pipeline import PipelineConfig, run_pipeline
from vision.storage import DriftlineRepository
from vision.types import ProgressSnapshot


class RuntimeTracker:
    def __init__(self) -> None:
        self._snapshots: dict[str, ProgressSnapshot] = {}
        self._lock = threading.Lock()

    def update(self, snapshot: ProgressSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.job_id] = snapshot

    def get(self, job_id: str) -> ProgressSnapshot | None:
        with self._lock:
            return self._snapshots.get(job_id)


def _write_upload(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _artifact_url(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return f"/artifacts/{relative_path}"


def _serialize_hazard(hazard: dict) -> dict:
    payload = dict(hazard)
    payload["thumbnail_url"] = _artifact_url(payload.get("thumbnail_path"))
    return payload


def _serialize_job_snapshot(payload: dict) -> dict:
    merged = dict(payload)
    merged["preview_image_url"] = _artifact_url(merged.get("preview_image_path") or merged.get("latest_preview_path"))
    return merged


def _serialize_dashboard(payload: dict) -> dict:
    return {
        "summary": payload["summary"],
        "recent_jobs": [
            _serialize_job_snapshot(job) for job in payload["recent_jobs"]
        ],
        "priority_repairs": [
            _serialize_hazard(hazard) for hazard in payload["priority_repairs"]
        ],
    }


def _build_job_snapshot(
    repository: DriftlineRepository,
    runtime: RuntimeTracker,
    job_id: str,
) -> dict | None:
    persisted = repository.get_job_status(job_id)
    if persisted is None:
        return None

    snapshot = runtime.get(job_id)
    merged = dict(persisted)
    if snapshot is not None:
        merged.update(asdict(snapshot))
    return _serialize_job_snapshot(merged)


def _spawn_job_thread(
    *,
    repository: DriftlineRepository,
    runtime: RuntimeTracker,
    config: PipelineConfig,
) -> None:
    def runner() -> None:
        try:
            runtime.update(
                ProgressSnapshot(
                    job_id=config.job_id,
                    status="queued",
                    frames_processed=0,
                    total_frames=0,
                    raw_detection_count=0,
                )
            )
            run_pipeline(config, progress_callback=runtime.update)
        except Exception as exc:
            existing = runtime.get(config.job_id)
            repository.mark_job_error(config.job_id, str(exc))
            runtime.update(
                ProgressSnapshot(
                    job_id=config.job_id,
                    status="error",
                    frames_processed=existing.frames_processed if existing else 0,
                    total_frames=existing.total_frames if existing else 0,
                    raw_detection_count=existing.raw_detection_count if existing else 0,
                    verified_hazard_count=existing.verified_hazard_count if existing else 0,
                    error=str(exc),
                    preview_image_path=existing.preview_image_path if existing else None,
                )
            )

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


def create_app(
    *,
    database_path: str | Path | None = None,
    upload_dir: str | Path | None = None,
    static_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    repo_root = Path(__file__).resolve().parent.parent
    resolved_upload_dir = Path(upload_dir or repo_root / "uploads")
    resolved_static_dir = Path(static_dir or Path(__file__).resolve().parent / "static")
    resolved_database_path = Path(database_path or repo_root / "runs" / "driftline.sqlite")
    resolved_artifacts_dir = Path(artifacts_dir or resolved_upload_dir / "artifacts")

    resolved_upload_dir.mkdir(parents=True, exist_ok=True)
    resolved_database_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)

    repository = DriftlineRepository(resolved_database_path)
    repository.initialize()
    runtime = RuntimeTracker()

    app.state.repository = repository
    app.state.runtime = runtime
    app.state.upload_dir = resolved_upload_dir
    app.state.static_dir = resolved_static_dir
    app.state.artifacts_dir = resolved_artifacts_dir
    app.state.default_model = repo_root / "vision" / "models" / "yolo12s_RDD2022_best.pt"

    app.mount("/static", StaticFiles(directory=str(resolved_static_dir)), name="static")
    app.mount("/artifacts", StaticFiles(directory=str(resolved_artifacts_dir)), name="artifacts")

    @app.get("/")
    async def dashboard_page() -> FileResponse:
        return FileResponse(resolved_static_dir / "index.html")

    @app.get("/community")
    async def community_page() -> FileResponse:
        return FileResponse(resolved_static_dir / "community.html")

    @app.get("/health")
    async def healthcheck() -> dict:
        return {"status": "ok"}

    @app.post("/api/jobs")
    async def create_job(
        video: UploadFile | None = File(None),
        gps_csv: UploadFile | None = File(None),
    ) -> dict:
        if video is None or not video.filename:
            raise HTTPException(status_code=400, detail="video is required")
        if gps_csv is None or not gps_csv.filename:
            raise HTTPException(status_code=400, detail="gps_csv is required")
        if not gps_csv.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="gps_csv must be a CSV file")

        job_id = str(uuid.uuid4())
        safe_video_name = Path(video.filename or "upload.mp4").name
        safe_gps_name = Path(gps_csv.filename or "track.csv").name
        video_path = resolved_upload_dir / f"{job_id}_{safe_video_name}"
        gps_path = resolved_upload_dir / f"{job_id}_{safe_gps_name}"

        _write_upload(video_path, await video.read())
        _write_upload(gps_path, await gps_csv.read())

        repository.create_job(
            job_id=job_id,
            video_filename=safe_video_name,
            gps_filename=safe_gps_name,
        )

        _spawn_job_thread(
            repository=repository,
            runtime=runtime,
            config=PipelineConfig(
                source=str(video_path),
                gps_csv=str(gps_path),
                database_path=str(resolved_database_path),
                job_id=job_id,
                model=str(app.state.default_model),
                artifacts_dir=str(resolved_artifacts_dir),
                enable_rerun=False,
            ),
        )
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        snapshot = _build_job_snapshot(repository, runtime, job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return snapshot

    @app.get("/api/jobs/{job_id}/events")
    async def get_job_events(job_id: str) -> StreamingResponse:
        if repository.get_job_status(job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")

        async def event_stream():
            while True:
                payload = _build_job_snapshot(repository, runtime, job_id)
                if payload is None:
                    break
                yield f"data: {json.dumps(payload)}\n\n"
                if payload["status"] in {"completed", "error"}:
                    break
                await asyncio.sleep(1.0)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/jobs/{job_id}/results")
    async def get_job_results(job_id: str) -> dict:
        results = repository.get_job_results(job_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job": _serialize_job_snapshot(results["job"]),
            "hazards": [_serialize_hazard(hazard) for hazard in results["hazards"]],
            "rebates": results["rebates"],
        }

    @app.get("/api/dashboard")
    async def get_dashboard() -> dict:
        return _serialize_dashboard(repository.get_dashboard_summary())

    @app.get("/api/community")
    async def get_community() -> dict:
        detections = repository.get_completed_raw_detections()
        clusters = [
            _serialize_hazard(asdict(cluster))
            for cluster in cluster_completed_jobs(detections)
            if cluster.verified
        ]
        dashboard_summary = repository.get_dashboard_summary()["summary"]
        return {
            "summary": {
                "completed_jobs": dashboard_summary["total_jobs"],
                "verified_hazard_count": len(clusters),
            },
            "hazards": clusters,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
