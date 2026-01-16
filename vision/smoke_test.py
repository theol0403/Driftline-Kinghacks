from __future__ import annotations

import argparse
import os

import cv2

from .detector import DetectorConfig, UltralyticsDetector
from .mapping import MappingConfig, OccupancyGridMapper
from .rerun_viz import RerunConfig, RerunLogger
from .types import Pose2D
from .vo import VisualOdometry, VisualOdometryConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for vision pipeline.")
    parser.add_argument("--video", type=str, required=True, help="Video path.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model path.")
    parser.add_argument("--frames", type=int, default=120, help="Frames to run.")
    parser.add_argument(
        "--rr-recording",
        type=str,
        default="smoke_test.rrd",
        help="Rerun recording output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {args.video}")

    detector = UltralyticsDetector(DetectorConfig(model_path=args.model))
    mapper = OccupancyGridMapper(MappingConfig())
    vo = VisualOdometry(VisualOdometryConfig())
    pose = Pose2D(0.0, 0.0, 0.0)

    rerun_logger = RerunLogger(RerunConfig(recording_path=args.rr_recording))

    for _ in range(args.frames):
        success, frame = cap.read()
        if not success:
            break
        detections = detector.detect(frame)
        pose_update = vo.update(frame)
        if pose_update is not None:
            pose = pose_update
        points = mapper.update(pose, detections, frame.shape)
        rerun_logger.log_frame(frame)
        rerun_logger.log_detections(detections)
        rerun_logger.log_pose(pose)
        rerun_logger.log_detection_points(points)
        rerun_logger.log_grid(mapper.grid_image())

    cap.release()


if __name__ == "__main__":
    main()
