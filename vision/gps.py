from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class GpsSample:
    time_s: float
    lat: float
    lon: float
    speed_mps: float
    heading_deg: float


class GpsTrack:
    def __init__(self, samples: Iterable[GpsSample]) -> None:
        ordered = sorted(samples, key=lambda sample: sample.time_s)
        self._samples: List[GpsSample] = ordered
        self._times: List[float] = [sample.time_s for sample in ordered]

    @property
    def samples(self) -> List[GpsSample]:
        return self._samples

    def nearest(self, time_s: float) -> Optional[GpsSample]:
        if not self._samples:
            return None
        index = bisect.bisect_left(self._times, time_s)
        if index == 0:
            return self._samples[0]
        if index >= len(self._samples):
            return self._samples[-1]
        before = self._samples[index - 1]
        after = self._samples[index]
        if (time_s - before.time_s) <= (after.time_s - time_s):
            return before
        return after


def load_gps_csv(path: str) -> GpsTrack:
    samples: List[GpsSample] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples.append(
                GpsSample(
                    time_s=float(row["time_s"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    speed_mps=float(row["speed_mps"]),
                    heading_deg=float(row["heading_deg"]),
                )
            )
    return GpsTrack(samples)
