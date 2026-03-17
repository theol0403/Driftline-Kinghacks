# Real Pipeline-Backed Hazard Dashboard

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `.agent/PLANS.md`.

## Purpose / Big Picture

After this change, a user can upload a dashcam video and a matching GPS CSV through the webapp, watch live processing progress, and see verified hazards plotted on a real Leaflet map instead of a mockup. The dashboard will rank the highest-priority hazards, estimate rebates, and expose a separate community map that merges hazards across all completed jobs. The same underlying processing path will also remain runnable from the command line through `python -m vision.main`.

## Progress

- [x] (2026-03-17 01:27Z) Explored the repository, read `ARCHITECTURE.md`, `webapp/app.py`, `vision/main.py`, `vision/mapping.py`, and `.agent/PLANS.md`.
- [x] (2026-03-17 01:31Z) Confirmed the current webapp launches a non-existent `src.pipeline` module and the checked-in `.venv` is missing required packages such as `fastapi` and `cv2`.
- [x] (2026-03-17 01:35Z) Locked product decisions for implementation: GPS CSV is required, verification requires 3 distinct frames, and the community map is a separate page.
- [x] (2026-03-17 02:02Z) Extracted the shared processing path into `vision/pipeline.py`, added SQLite persistence in `vision/storage.py`, and added DBSCAN clustering/scoring in `vision/clustering.py`.
- [x] (2026-03-17 02:12Z) Replaced the FastAPI mock flow with real background processing, `/api/...` routes, SSE progress, and static route handling in `webapp/app.py`.
- [x] (2026-03-17 02:20Z) Rebuilt the frontend into a real Leaflet dashboard plus separate community page backed by the new APIs.
- [x] (2026-03-17 02:26Z) Added automated tests for projection, clustering, severity scoring, and API behavior under `tests/`.
- [x] (2026-03-17 02:31Z) Installed dependencies into the repo `.venv`, ran `./.venv/bin/python -m pytest`, and verified the CLI and FastAPI app import paths.
- [x] (2026-03-17 02:52Z) Added persisted artifact output for live annotated preview frames and per-detection crop thumbnails.
- [x] (2026-03-17 02:57Z) Updated the dashboard and community map to show hover thumbnails on waypoints and image-backed hazard cards in the scrollable sidebar.

## Surprises & Discoveries

- Observation: `webapp/app.py` still shells out to `python -m src.pipeline`, but no `src/` package exists in the repository.
  Evidence: `rg --files` found `vision/main.py` as the real entrypoint and no `src.pipeline`.

- Observation: The checked-in `.venv` does not currently contain the runtime dependencies needed for either the vision CLI or the FastAPI app.
  Evidence: `./.venv/bin/python -m vision.main --help` failed with `ModuleNotFoundError: No module named 'cv2'`, and `./.venv/bin/python -c "import fastapi"` failed with `ModuleNotFoundError: No module named 'fastapi'`.

- Observation: The existing frontend is a pure static mockup with placeholder counters and a placeholder map image.
  Evidence: `webapp/static/script.js` adds fake counts and `webapp/static/index.html` embeds a placeholder image instead of a real map.

- Observation: FastAPI testing in this repo requires `httpx`, which is not pulled in by the runtime-only dependency set.
  Evidence: the first `pytest` run failed during `fastapi.testclient.TestClient` import with `RuntimeError: The starlette.testclient module requires the httpx package to be installed.`

- Observation: Image-backed hazard interaction needs one shared artifact path, otherwise the live preview, map hover tooltip, and sidebar cards drift apart.
  Evidence: before this pass the pipeline persisted only numeric detections and clusters, leaving the frontend with no reusable image source for any of the requested interactions.

## Decision Log

- Decision: Keep one canonical processing path by extracting a shared `run_pipeline()` function and making the CLI a thin wrapper around it.
  Rationale: The repo currently diverges between the real vision code and a dead webapp subprocess target. A shared entrypoint removes that drift.
  Date/Author: 2026-03-17 / Codex

- Decision: Require a GPS CSV upload for every web job.
  Rationale: The dashboard’s core promise is defensible geolocated hazards. Falling back to demo tracks would make the map visually impressive but technically misleading.
  Date/Author: 2026-03-17 / Codex

- Decision: Define a verified hazard as a same-category DBSCAN cluster with at least 3 detections from distinct frames.
  Rationale: This matches the user’s pitch language and gives a simple, conservative false-positive story for the demo.
  Date/Author: 2026-03-17 / Codex

- Decision: Build the community view as a separate page.
  Rationale: The per-job dashboard and city-wide aggregation have different jobs to do. Splitting them avoids overloading one screen.
  Date/Author: 2026-03-17 / Codex

- Decision: Keep the frontend static-file based and use Leaflet via CDN instead of introducing a build tool or SPA framework.
  Rationale: The existing webapp is already static HTML/CSS/JS, and the requested features fit within that architecture without adding extra hackathon risk.
  Date/Author: 2026-03-17 / Codex

- Decision: Persist annotated preview frames and crop thumbnails as files under an artifact directory, and expose them through a mounted static route.
  Rationale: File-backed artifacts are simple to refresh live during processing, trivial to reuse across API payloads, and avoid storing binary blobs in SQLite for a demo-oriented workflow.
  Date/Author: 2026-03-17 / Codex

## Outcomes & Retrospective

The repository now has a working upload-to-map flow backed by real pipeline results instead of mocks. `vision/main.py` delegates to `vision/pipeline.py`, SQLite persistence lives in `vision/storage.py`, DBSCAN verification and severity scoring live in `vision/clustering.py`, and live/hover imagery is persisted through the artifact path used by `vision/artifacts.py`. The webapp now exposes real `/api/...` routes, a live SSE progress feed, a live preview frame, image-backed waypoint hover cards, a scrollable hazard gallery, and a separate community map page.

Validation is complete for the coded scope. `./.venv/bin/python -m pytest` passes with six tests covering projection, verification behavior, severity recency, dashboard ordering, and cross-job aggregation. The remaining manual step is to run the server and upload real footage for a browser demo.

## Context and Orientation

The repository has two main subsystems relevant to this work. The `vision/` package contains the detection pipeline. `vision/main.py` currently owns the control flow: open the video, run the detector, optionally update visual odometry, look up GPS samples, and send visualization data to Rerun. `vision/mapping.py` currently estimates local ground-plane points for detections but does not persist anything. `vision/gps.py` loads the sidecar CSV and provides nearest-sample lookup by frame timestamp.

The `webapp/` package contains a FastAPI server and static frontend files. `webapp/app.py` accepts a video upload, stores a tiny in-memory job dict, and tries to launch a non-existent `src.pipeline` subprocess. `webapp/static/index.html`, `webapp/static/style.css`, and `webapp/static/script.js` render a polished mock dashboard, but the map, summary numbers, and activity entries are still hard-coded.

SQLite is the embedded database engine already referenced by the webapp. DBSCAN is a density-based clustering algorithm from scikit-learn. In this repository it will group nearby detections in projected meter coordinates so that repeated observations of the same real-world hazard become one hazard event.

## Plan of Work

Create a new processing module in `vision/` that owns the end-to-end job flow. It should accept a typed configuration object, open the video, compute frame timestamps, run detection, load nearest GPS samples, project detections into hazard lat/lon points, persist raw detections, and periodically call a progress callback. The existing CLI in `vision/main.py` should delegate to this function and keep Rerun logging optional.

Add a thin SQLite repository layer under `vision/` so the rest of the code does not issue SQL ad hoc. The repository should create the `jobs`, `raw_detections`, and `hazard_clusters` tables on startup, insert job rows, append raw detections, replace per-job clusters, and query dashboard/community summaries.

Add a clustering module under `vision/` that converts lat/lon to local meter coordinates, groups detections by job and category with `sklearn.cluster.DBSCAN`, marks clusters with fewer than three distinct frames as unverified, and computes centroid, confidence, first/last seen, and severity score. Reuse that module for both per-job clustering and cross-job community aggregation.

Refactor `webapp/app.py` to own one SQLite database file in `runs/driftline.sqlite`, start background threads that call the shared pipeline function directly, expose JSON APIs under `/api/...`, and expose a progress stream via FastAPI `StreamingResponse` using the `text/event-stream` media type. Keep in-memory progress state only for live updates during the current process.

Replace the frontend with real data-driven pages. The dashboard should support selecting both a video and a GPS CSV, show a progress bar during processing, render verified hazards on a Leaflet map, display a priority-repairs list from the highest severity scores, and calculate rebates from distance plus verified hazard count. Add a second static HTML page for `/community` that loads the aggregated map data.

Add tests under `tests/` for the geospatial helpers, clustering rules, severity scoring, and FastAPI endpoints. Use small synthetic inputs so the tests stay fast and avoid the heavy detector path.

## Concrete Steps

Work from the repository root:

    cd /Users/theol/Documents/github/Driftline-Kinghacks

During implementation, use the repo virtual environment:

    ./.venv/bin/python -m pytest
    ./.venv/bin/python -m uvicorn webapp.app:app --reload

If dependencies are missing, install them into the same `.venv` before validation:

    uv pip install -r requirements.txt

When the server is running, open the dashboard at `http://127.0.0.1:8000/` and the community page at `http://127.0.0.1:8000/community`.

## Validation and Acceptance

Acceptance is behavioral. A successful implementation lets a user upload one video and one matching GPS CSV, watch the progress bar advance while the server processes frames, and then see verified hazards appear as colored map markers on the dashboard. The “Priority Repairs” panel must show the top ten verified hazards sorted by severity. The “Rebates” panel must reflect the uploaded route distance and verified hazard count.

The API acceptance checks are:

1. `POST /api/jobs` rejects requests that omit the GPS CSV.
2. `GET /api/jobs/{job_id}` reports live `frames_processed`, `total_frames`, and final status.
3. `GET /api/jobs/{job_id}/results` returns verified hazards and the rebate summary for that job.
4. `GET /api/dashboard` returns recent jobs plus the top ten priority repairs.
5. `GET /api/community` returns verified aggregated hazards merged across completed jobs.

The automated validation path is to run `./.venv/bin/python -m pytest` and expect the new tests to pass. Manual validation then confirms the upload flow and the map views in a browser.

## Idempotence and Recovery

Creating the database schema must be idempotent: the code should use `CREATE TABLE IF NOT EXISTS` and upsert or replace data for the current job rather than assuming a fresh database. Re-running a job with the same `job_id` is not required, but starting new jobs repeatedly must not corrupt prior results. If a job fails partway through processing, the server should mark it as `error`, preserve the error message on the `jobs` row, and leave any existing completed jobs queryable.

## Artifacts and Notes

The most important proof points after implementation are:

    POST /api/jobs with both files returns {"job_id": "..."}.
    GET /api/jobs/<job_id> transitions from queued -> processing -> completed.
    GET /api/jobs/<job_id>/results returns verified hazards with centroid_lat, centroid_lon, detection_count, avg_confidence, and severity_score.
    GET /api/community returns aggregated verified hazards across multiple jobs.

## Interfaces and Dependencies

In `vision/pipeline.py`, define a typed configuration and shared entrypoint:

    @dataclass
    class PipelineConfig:
        source: str
        gps_csv: str
        database_path: str
        job_id: str
        model: str
        conf: float = 0.25
        imgsz: int = 640
        categories: Optional[list[str]] = None
        no_vo: bool = False
        vo_scale: float = 0.02
        map_width: float = 50.0
        map_height: float = 50.0
        map_resolution: float = 0.2
        rerun_recording: Optional[str] = None
        enable_rerun: bool = False

    def run_pipeline(
        config: PipelineConfig,
        progress_callback: Optional[Callable[[ProgressSnapshot], None]] = None,
    ) -> JobResult:
        ...

In `vision/storage.py`, provide a repository API that owns schema creation and queries:

    class DriftlineRepository:
        def initialize(self) -> None: ...
        def create_job(...): ...
        def update_job_progress(...): ...
        def mark_job_completed(...): ...
        def mark_job_error(...): ...
        def insert_raw_detections(...): ...
        def replace_job_clusters(...): ...
        def get_job_results(job_id: str) -> dict: ...
        def get_dashboard_summary() -> dict: ...
        def get_community_clusters() -> list[dict]: ...

In `vision/clustering.py`, provide helpers for coordinate projection, clustering, and scoring:

    def cluster_projected_detections(
        detections: Sequence[ProjectedDetection],
        eps_m: float = 2.5,
        min_samples: int = 3,
    ) -> list[HazardCluster]:
        ...

Revision note: Created this ExecPlan at implementation start because `AGENTS.md` requires an ExecPlan for complex features and the repo had no existing execution plan for this work.

Revision note: Updated this ExecPlan after implementation to capture the finished modules, the frontend/backend/API changes, the dependency discovery around `httpx`, the artifact-backed live preview and thumbnail pass, and the final validation results.
