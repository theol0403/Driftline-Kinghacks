from __future__ import annotations

import argparse
import os
import uuid

from .pipeline import PipelineConfig, run_pipeline
from .storage import DriftlineRepository


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
        default=os.path.join(
            os.path.dirname(__file__), "models", "yolo12s_RDD2022_best.pt"
        ),
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
        help="Filter detections to these categories (e.g. pothole).",
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
        "--no-rerun",
        action="store_true",
        help="Disable the live Rerun viewer.",
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
    parser.add_argument(
        "--database-path",
        type=str,
        default=os.path.join("runs", "cli.sqlite"),
        help="SQLite path for persisted detections and hazard clusters.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Optional explicit job identifier.",
    )
    return parser.parse_args()


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

    if not args.gps_csv:
        raise ValueError("--gps-csv is required")

    job_id = args.job_id or str(uuid.uuid4())
    repository = DriftlineRepository(args.database_path)
    repository.initialize()
    repository.create_job(
        job_id=job_id,
        video_filename=os.path.basename(source),
        gps_filename=os.path.basename(args.gps_csv),
    )

    run_pipeline(
        PipelineConfig(
            source=source,
            gps_csv=args.gps_csv,
            database_path=args.database_path,
            job_id=job_id,
            model=args.model,
            artifacts_dir=os.path.join("runs", "artifacts"),
            conf=args.conf,
            imgsz=args.imgsz,
            categories=args.categories,
            no_vo=args.no_vo,
            vo_scale=args.vo_scale,
            map_width=args.map_width,
            map_height=args.map_height,
            map_resolution=args.map_resolution,
            label_map_path=args.label_map,
            rerun_recording=args.rr_recording,
            enable_rerun=not args.no_rerun,
        )
    )


if __name__ == "__main__":
    main()
