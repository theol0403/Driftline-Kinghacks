from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Optional

import cv2
import rerun as rr

from .detector import DetectorConfig, UltralyticsDetector, filter_by_category
from .gps import GpsTrack, load_gps_csv
from .mapping import MappingConfig, OccupancyGridMapper
from .rerun_viz import RerunConfig, RerunLogger
from .types import Pose2D
from .vo import VisualOdometry, VisualOdometryConfig


def load_label_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime Kingston vision mapping pipeline."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index or video path.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo video from vision/demo_assets.",
    )
    parser.add_argument(
        "--demo-video",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "demo_assets", "demo.mp4"),
        help="Path to demo video.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model path or name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Optional JSON mapping of model labels to categories.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Filter detections to these categories (e.g. potholes).",
    )
    parser.add_argument(
        "--no-vo",
        action="store_true",
        help="Disable visual odometry (map stays fixed).",
    )
    parser.add_argument(
        "--vo-scale",
        type=float,
        default=0.02,
        help="Meters per pixel scale for VO translation.",
    )
    parser.add_argument(
        "--map-width",
        type=float,
        default=50.0,
        help="Map width in meters.",
    )
    parser.add_argument(
        "--map-height",
        type=float,
        default=50.0,
        help="Map height in meters.",
    )
    parser.add_argument(
        "--map-resolution",
        type=float,
        default=0.2,
        help="Map resolution in meters per cell.",
    )
    parser.add_argument(
        "--rr-recording",
        type=str,
        default=None,
        help="Optional path to save a Rerun recording (.rrd).",
    )
    parser.add_argument(
        "--gps-csv",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "demo_assets",
            "kingston_test",
            "street_view_5m.csv",
        ),
        help="CSV with time_s,lat,lon,speed_mps,heading_deg for map logging.",
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> None:
    args = parse_args()
    if args.demo:
        source = args.demo_video
        if not os.path.exists(source):
            raise FileNotFoundError(
                f"Demo video not found at {source}. Add one to vision/demo_assets/."
            )
    else:
        source = args.source

    label_map = load_label_map(args.label_map)
    detector = UltralyticsDetector(
        DetectorConfig(
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            label_map=label_map,
        )
    )

    mapper = OccupancyGridMapper(
        MappingConfig(
            width_m=args.map_width,
            height_m=args.map_height,
            resolution_m=args.map_resolution,
        )
    )

    vo = None
    pose = Pose2D(0.0, 0.0, 0.0)
    if not args.no_vo:
        vo = VisualOdometry(VisualOdometryConfig(scale_m_per_px=args.vo_scale))

    rerun_logger = RerunLogger(RerunConfig(recording_path=args.rr_recording))

    gps_track: Optional[GpsTrack] = None
    if args.gps_csv:
        if not os.path.exists(args.gps_csv):
            raise FileNotFoundError(f"GPS CSV not found at {args.gps_csv}")
        gps_track = load_gps_csv(args.gps_csv)

    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_index += 1

        t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_s <= 0.0 and fps > 0.0:
            t_s = frame_index / fps
        rr.set_time("time", duration=t_s)

        detections = detector.detect(frame)
        detections = filter_by_category(detections, args.categories)

        if vo is not None:
            pose_update = vo.update(frame)
            if pose_update is not None:
                pose = pose_update

        points = mapper.update(pose, detections, frame.shape)
        grid = mapper.grid_image()

        rerun_logger.log_frame(frame)
        rerun_logger.log_detections(detections)
        rerun_logger.log_pose(pose)
        if gps_track is not None:
            sample = gps_track.nearest(t_s)
            if sample is not None:
                rerun_logger.log_gps(sample.lat, sample.lon)
        rerun_logger.log_detection_points(points)
        rerun_logger.log_grid(grid)

    cap.release()


if __name__ == "__main__":
    main()
