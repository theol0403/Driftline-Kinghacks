from __future__ import annotations

import argparse
import csv
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class GsvConfig:
    api_key: str
    size: tuple[int, int]
    scale: int
    fov: int
    pitch: int
    step_m: float
    fps: int
    sleep_s: float
    output_dir: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smooth Street View video from a GPS route.",
    )
    parser.add_argument("--api_key", required=True, help="Google Maps API key.")
    parser.add_argument("--route_csv", required=True, help="CSV with lat,lon per row.")
    parser.add_argument("--output_mp4", required=True, help="Output video path.")
    parser.add_argument(
        "--output_gps_csv",
        default="",
        help="Optional GPS CSV with one row per frame.",
    )
    parser.add_argument("--size", default="640x640", help="Image size, e.g. 640x640.")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (1 or 2).")
    parser.add_argument("--fov", type=int, default=90, help="Field of view.")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch angle.")
    parser.add_argument("--step_m", type=float, default=2.0, help="Meters per frame.")
    parser.add_argument("--fps", type=int, default=30, help="Output video fps.")
    parser.add_argument(
        "--sleep_s",
        type=float,
        default=0.05,
        help="Sleep between requests to avoid throttling.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        help="Optional cache directory for downloaded frames.",
    )
    return parser.parse_args()


def parse_size(size_str: str) -> tuple[int, int]:
    try:
        w_str, h_str = size_str.lower().split("x")
        return int(w_str), int(h_str)
    except ValueError as exc:
        raise ValueError(f"Invalid --size format: {size_str}") from exc


def haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def interpolate_route(
    points: list[tuple[float, float]],
    step_m: float,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points

    dense: list[tuple[float, float]] = [points[0]]
    for i in range(1, len(points)):
        p1, p2 = points[i - 1], points[i]
        dist = haversine_m(p1, p2)
        if dist <= step_m:
            dense.append(p2)
            continue
        steps = max(1, int(dist // step_m))
        lat1, lon1 = p1
        lat2, lon2 = p2
        for s in range(1, steps + 1):
            t = s / (steps + 1)
            lat = lat1 + (lat2 - lat1) * t
            lon = lon1 + (lon2 - lon1) * t
            dense.append((lat, lon))
        dense.append(p2)
    return dense


def load_route_csv(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            lat = float(row[0].strip())
            lon = float(row[1].strip())
            points.append((lat, lon))
    if len(points) < 2:
        raise ValueError("Route must contain at least two points.")
    return points


def gsv_url(
    lat: float,
    lon: float,
    heading: float,
    config: GsvConfig,
) -> str:
    params = {
        "size": f"{config.size[0]}x{config.size[1]}",
        "scale": str(config.scale),
        "location": f"{lat:.6f},{lon:.6f}",
        "heading": f"{heading:.2f}",
        "pitch": str(config.pitch),
        "fov": str(config.fov),
        "key": config.api_key,
    }
    return "https://maps.googleapis.com/maps/api/streetview?" + urllib.parse.urlencode(params)


def fetch_image(
    url: str,
    cache_path: Path | None,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        data = cache_path.read_bytes()
    else:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode image from Street View response.")
    return image


def write_gps_csv(path: Path, rows: list[tuple[float, float, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "heading_deg"])
        writer.writerows(rows)


def build_video(
    route: list[tuple[float, float]],
    config: GsvConfig,
    output_mp4: Path,
    output_gps_csv: Path | None,
) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_mp4),
        fourcc,
        config.fps,
        config.size,
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer.")

    gps_rows: list[tuple[float, float, float]] = []
    try:
        for i in range(len(route)):
            p = route[i]
            p_next = route[min(i + 1, len(route) - 1)]
            heading = bearing_deg(p, p_next)
            url = gsv_url(p[0], p[1], heading, config)
            cache_path = None
            if config.output_dir:
                cache_path = config.output_dir / f"frame_{i:05d}.jpg"
            image = fetch_image(url, cache_path)
            if (image.shape[1], image.shape[0]) != config.size:
                image = cv2.resize(image, config.size, interpolation=cv2.INTER_AREA)
            writer.write(image)
            gps_rows.append((p[0], p[1], heading))
            if config.sleep_s > 0:
                time.sleep(config.sleep_s)
    finally:
        writer.release()

    if output_gps_csv:
        write_gps_csv(output_gps_csv, gps_rows)


def main() -> None:
    args = parse_args()
    size = parse_size(args.size)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    config = GsvConfig(
        api_key=args.api_key,
        size=size,
        scale=args.scale,
        fov=args.fov,
        pitch=args.pitch,
        step_m=args.step_m,
        fps=args.fps,
        sleep_s=args.sleep_s,
        output_dir=cache_dir,
    )
    route = load_route_csv(Path(args.route_csv))
    dense_route = interpolate_route(route, config.step_m)

    output_mp4 = Path(args.output_mp4)
    output_gps_csv = Path(args.output_gps_csv) if args.output_gps_csv else None
    build_video(dense_route, config, output_mp4, output_gps_csv)


if __name__ == "__main__":
    main()
