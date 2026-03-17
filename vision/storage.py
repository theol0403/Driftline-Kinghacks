from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from .types import HazardCluster, ProjectedDetection


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class DriftlineRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {row["name"] for row in rows}
        if column_name not in existing:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    video_filename TEXT NOT NULL,
                    gps_filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    frames_processed INTEGER NOT NULL DEFAULT 0,
                    total_frames INTEGER NOT NULL DEFAULT 0,
                    raw_detection_count INTEGER NOT NULL DEFAULT 0,
                    verified_hazard_count INTEGER NOT NULL DEFAULT 0,
                    distance_km REAL NOT NULL DEFAULT 0,
                    credits_earned REAL NOT NULL DEFAULT 0,
                    latest_preview_path TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS raw_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    frame_ts REAL NOT NULL,
                    category TEXT NOT NULL,
                    label TEXT NOT NULL,
                    score REAL NOT NULL,
                    vehicle_lat REAL NOT NULL,
                    vehicle_lon REAL NOT NULL,
                    vehicle_heading_deg REAL NOT NULL,
                    hazard_lat REAL NOT NULL,
                    hazard_lon REAL NOT NULL,
                    offset_forward_m REAL NOT NULL,
                    offset_lateral_m REAL NOT NULL,
                    local_x_m REAL NOT NULL,
                    local_y_m REAL NOT NULL,
                    crop_path TEXT,
                    observed_at_iso TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS hazard_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    centroid_lat REAL NOT NULL,
                    centroid_lon REAL NOT NULL,
                    detection_count INTEGER NOT NULL,
                    unique_frame_count INTEGER NOT NULL,
                    avg_confidence REAL NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    severity_score REAL NOT NULL,
                    verified INTEGER NOT NULL,
                    observed_at_iso TEXT NOT NULL,
                    thumbnail_path TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column(connection, "jobs", "latest_preview_path", "TEXT")
            self._ensure_column(connection, "raw_detections", "crop_path", "TEXT")
            self._ensure_column(connection, "hazard_clusters", "thumbnail_path", "TEXT")
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_raw_detections_job_id ON raw_detections(job_id);
                CREATE INDEX IF NOT EXISTS idx_raw_detections_category ON raw_detections(category);
                CREATE INDEX IF NOT EXISTS idx_hazard_clusters_job_id ON hazard_clusters(job_id);
                CREATE INDEX IF NOT EXISTS idx_hazard_clusters_verified ON hazard_clusters(verified);
                CREATE INDEX IF NOT EXISTS idx_hazard_clusters_severity ON hazard_clusters(severity_score DESC);
                """
            )

    def create_job(
        self,
        *,
        job_id: str,
        video_filename: str,
        gps_filename: str,
        status: str = "queued",
        created_at: str | None = None,
    ) -> None:
        timestamp = created_at or utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id,
                    video_filename,
                    gps_filename,
                    status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, video_filename, gps_filename, status, timestamp),
            )

    def update_job_progress(
        self,
        job_id: str,
        *,
        status: str | None = None,
        frames_processed: int | None = None,
        total_frames: int | None = None,
        raw_detection_count: int | None = None,
        verified_hazard_count: int | None = None,
        error: str | None = None,
        latest_preview_path: str | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[object] = []
        for field, value in (
            ("status", status),
            ("frames_processed", frames_processed),
            ("total_frames", total_frames),
            ("raw_detection_count", raw_detection_count),
            ("verified_hazard_count", verified_hazard_count),
            ("error", error),
            ("latest_preview_path", latest_preview_path),
        ):
            if value is not None:
                assignments.append(f"{field} = ?")
                values.append(value)

        if not assignments:
            return

        values.append(job_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ?",
                values,
            )

    def mark_job_completed(
        self,
        job_id: str,
        *,
        frames_processed: int,
        total_frames: int,
        raw_detection_count: int,
        verified_hazard_count: int,
        distance_km: float,
        credits_earned: float,
        latest_preview_path: str | None = None,
        completed_at: str | None = None,
    ) -> str:
        completion_time = completed_at or utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?,
                    frames_processed = ?,
                    total_frames = ?,
                    raw_detection_count = ?,
                    verified_hazard_count = ?,
                    distance_km = ?,
                    credits_earned = ?,
                    latest_preview_path = COALESCE(?, latest_preview_path),
                    completed_at = ?,
                    error = NULL
                WHERE job_id = ?
                """,
                (
                    "completed",
                    frames_processed,
                    total_frames,
                    raw_detection_count,
                    verified_hazard_count,
                    distance_km,
                    credits_earned,
                    latest_preview_path,
                    completion_time,
                    job_id,
                ),
            )
        return completion_time

    def mark_job_error(self, job_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, error = ?
                WHERE job_id = ?
                """,
                ("error", error, job_id),
            )

    def insert_raw_detections(
        self,
        detections: Sequence[ProjectedDetection],
    ) -> None:
        if not detections:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO raw_detections (
                    job_id,
                    frame_index,
                    frame_ts,
                    category,
                    label,
                    score,
                    vehicle_lat,
                    vehicle_lon,
                    vehicle_heading_deg,
                    hazard_lat,
                    hazard_lon,
                    offset_forward_m,
                    offset_lateral_m,
                    local_x_m,
                    local_y_m,
                    crop_path,
                    observed_at_iso
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        detection.job_id,
                        detection.frame_index,
                        detection.frame_ts,
                        detection.category,
                        detection.label,
                        detection.score,
                        detection.vehicle_lat,
                        detection.vehicle_lon,
                        detection.vehicle_heading_deg,
                        detection.hazard_lat,
                        detection.hazard_lon,
                        detection.offset_forward_m,
                        detection.offset_lateral_m,
                        detection.local_x_m,
                        detection.local_y_m,
                        detection.crop_path,
                        detection.observed_at_iso,
                    )
                    for detection in detections
                ],
            )

    def replace_job_clusters(
        self,
        job_id: str,
        clusters: Sequence[HazardCluster],
    ) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM hazard_clusters WHERE job_id = ?", (job_id,))
            connection.executemany(
                """
                INSERT INTO hazard_clusters (
                    job_id,
                    category,
                    centroid_lat,
                    centroid_lon,
                    detection_count,
                    unique_frame_count,
                    avg_confidence,
                    first_seen,
                    last_seen,
                    severity_score,
                    verified,
                    observed_at_iso,
                    thumbnail_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        cluster.category,
                        cluster.centroid_lat,
                        cluster.centroid_lon,
                        cluster.detection_count,
                        cluster.unique_frame_count,
                        cluster.avg_confidence,
                        cluster.first_seen,
                        cluster.last_seen,
                        cluster.severity_score,
                        int(cluster.verified),
                        cluster.observed_at_iso,
                        cluster.thumbnail_path,
                    )
                    for cluster in clusters
                ],
            )

    def get_job_status(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    job_id,
                    video_filename,
                    gps_filename,
                    status,
                    frames_processed,
                    total_frames,
                    raw_detection_count,
                    verified_hazard_count,
                    distance_km,
                    credits_earned,
                    latest_preview_path,
                    created_at,
                    completed_at,
                    error
                FROM jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_job_results(self, job_id: str) -> dict | None:
        job = self.get_job_status(job_id)
        if job is None:
            return None

        with self._connect() as connection:
            hazards = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT
                        category,
                        centroid_lat,
                        centroid_lon,
                        detection_count,
                        unique_frame_count,
                        avg_confidence,
                        first_seen,
                        last_seen,
                        severity_score,
                        verified,
                        thumbnail_path
                    FROM hazard_clusters
                    WHERE job_id = ? AND verified = 1
                    ORDER BY severity_score DESC, last_seen DESC
                    """,
                    (job_id,),
                ).fetchall()
            ]

        return {
            "job": job,
            "hazards": hazards,
            "rebates": {
                "distance_km": job["distance_km"],
                "verified_hazard_count": job["verified_hazard_count"],
                "credits_earned": job["credits_earned"],
            },
        }

    def get_dashboard_summary(self) -> dict:
        with self._connect() as connection:
            summary_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_jobs,
                    COALESCE(SUM(credits_earned), 0.0) AS total_credits_earned,
                    COALESCE(SUM(verified_hazard_count), 0) AS total_verified_hazards,
                    COALESCE(SUM(distance_km), 0.0) AS total_distance_km
                FROM jobs
                WHERE status = 'completed'
                """
            ).fetchone()

            recent_jobs = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT
                        job_id,
                        video_filename,
                        gps_filename,
                        status,
                        verified_hazard_count,
                        raw_detection_count,
                        distance_km,
                        credits_earned,
                        latest_preview_path,
                        created_at,
                        completed_at
                    FROM jobs
                    ORDER BY created_at DESC
                    LIMIT 10
                    """
                ).fetchall()
            ]

            priority_repairs = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT
                        hc.job_id,
                        j.video_filename,
                        hc.category,
                        hc.centroid_lat,
                        hc.centroid_lon,
                        hc.detection_count,
                        hc.unique_frame_count,
                        hc.avg_confidence,
                        hc.first_seen,
                        hc.last_seen,
                        hc.severity_score,
                        hc.observed_at_iso,
                        hc.thumbnail_path
                    FROM hazard_clusters hc
                    JOIN jobs j ON j.job_id = hc.job_id
                    WHERE j.status = 'completed' AND hc.verified = 1
                    ORDER BY hc.severity_score DESC, hc.last_seen DESC
                    LIMIT 10
                    """
                ).fetchall()
            ]

        return {
            "summary": dict(summary_row),
            "recent_jobs": recent_jobs,
            "priority_repairs": priority_repairs,
        }

    def get_completed_raw_detections(self) -> list[ProjectedDetection]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    rd.job_id,
                    rd.frame_index,
                    rd.frame_ts,
                    rd.category,
                    rd.label,
                    rd.score,
                    rd.vehicle_lat,
                    rd.vehicle_lon,
                    rd.vehicle_heading_deg,
                    rd.hazard_lat,
                    rd.hazard_lon,
                    rd.offset_forward_m,
                    rd.offset_lateral_m,
                    rd.local_x_m,
                    rd.local_y_m,
                    rd.crop_path,
                    COALESCE(j.completed_at, rd.observed_at_iso, j.created_at) AS observed_at_iso
                FROM raw_detections rd
                JOIN jobs j ON j.job_id = rd.job_id
                WHERE j.status = 'completed'
                """
            ).fetchall()

        return [
            ProjectedDetection(
                job_id=row["job_id"],
                frame_index=row["frame_index"],
                frame_ts=row["frame_ts"],
                category=row["category"],
                label=row["label"],
                score=row["score"],
                vehicle_lat=row["vehicle_lat"],
                vehicle_lon=row["vehicle_lon"],
                vehicle_heading_deg=row["vehicle_heading_deg"],
                hazard_lat=row["hazard_lat"],
                hazard_lon=row["hazard_lon"],
                offset_forward_m=row["offset_forward_m"],
                offset_lateral_m=row["offset_lateral_m"],
                local_x_m=row["local_x_m"],
                local_y_m=row["local_y_m"],
                crop_path=row["crop_path"],
                observed_at_iso=row["observed_at_iso"],
            )
            for row in rows
        ]
