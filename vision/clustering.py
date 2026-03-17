from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import DBSCAN

from .mapping import latlon_to_local_xy
from .types import HazardCluster, ProjectedDetection


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def recency_weight(observed_at_iso: str | None, now: datetime | None = None) -> float:
    reference = now or datetime.now(UTC)
    age_days = max(
        (reference - parse_iso_datetime(observed_at_iso)).total_seconds() / 86_400.0,
        0.0,
    )
    return math.exp(-age_days / 7.0)


def compute_severity_score(
    detection_count: int,
    avg_confidence: float,
    observed_at_iso: str | None,
    now: datetime | None = None,
) -> float:
    return detection_count * avg_confidence * recency_weight(observed_at_iso, now)


def _build_cluster(
    category: str,
    members: Sequence[ProjectedDetection],
    observed_at_iso: str,
    now: datetime,
) -> HazardCluster:
    detection_count = len(members)
    unique_frames = len({(item.job_id, item.frame_index) for item in members})
    avg_confidence = float(np.mean([item.score for item in members]))
    centroid_lat = float(np.mean([item.hazard_lat for item in members]))
    centroid_lon = float(np.mean([item.hazard_lon for item in members]))
    first_seen = float(min(item.frame_ts for item in members))
    last_seen = float(max(item.frame_ts for item in members))
    verified = unique_frames >= 3
    best_member = max(members, key=lambda item: item.score)
    severity_score = compute_severity_score(
        detection_count=detection_count,
        avg_confidence=avg_confidence,
        observed_at_iso=observed_at_iso,
        now=now,
    )
    return HazardCluster(
        job_id=members[0].job_id if len({item.job_id for item in members}) == 1 else None,
        category=category,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        detection_count=detection_count,
        unique_frame_count=unique_frames,
        avg_confidence=avg_confidence,
        first_seen=first_seen,
        last_seen=last_seen,
        severity_score=severity_score,
        verified=verified,
        observed_at_iso=observed_at_iso,
        thumbnail_path=best_member.crop_path,
    )


def cluster_projected_detections(
    detections: Sequence[ProjectedDetection],
    eps_m: float = 2.5,
    min_samples: int = 3,
    now: datetime | None = None,
) -> list[HazardCluster]:
    if not detections:
        return []

    reference_now = now or datetime.now(UTC)
    grouped: dict[str, list[ProjectedDetection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.category].append(detection)

    clusters: list[HazardCluster] = []
    for category, items in grouped.items():
        origin_lat = float(np.mean([item.hazard_lat for item in items]))
        origin_lon = float(np.mean([item.hazard_lon for item in items]))
        coords = np.array(
            [
                latlon_to_local_xy(
                    item.hazard_lat,
                    item.hazard_lon,
                    origin_lat,
                    origin_lon,
                )
                for item in items
            ]
        )
        labels = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean").fit_predict(coords)

        indexed_members: dict[int, list[ProjectedDetection]] = defaultdict(list)
        noise_members: list[ProjectedDetection] = []
        for label, item in zip(labels.tolist(), items):
            if label == -1:
                noise_members.append(item)
            else:
                indexed_members[int(label)].append(item)

        for _, members in sorted(indexed_members.items()):
            observed_at_iso = max(
                (member.observed_at_iso or iso_now()) for member in members
            )
            clusters.append(
                _build_cluster(
                    category=category,
                    members=members,
                    observed_at_iso=observed_at_iso,
                    now=reference_now,
                )
            )

        for member in noise_members:
            observed_at_iso = member.observed_at_iso or iso_now()
            clusters.append(
                _build_cluster(
                    category=category,
                    members=[member],
                    observed_at_iso=observed_at_iso,
                    now=reference_now,
                )
            )

    clusters.sort(key=lambda item: item.severity_score, reverse=True)
    return clusters


def cluster_completed_jobs(
    detections: Iterable[ProjectedDetection],
    eps_m: float = 2.5,
    min_samples: int = 3,
    now: datetime | None = None,
) -> list[HazardCluster]:
    return cluster_projected_detections(
        list(detections),
        eps_m=eps_m,
        min_samples=min_samples,
        now=now,
    )
