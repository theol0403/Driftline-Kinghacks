from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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
