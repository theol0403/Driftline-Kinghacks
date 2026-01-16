## Driftline Kingston Vision

Robotic vision pipeline for Kingston public-works insights. The system ingests vehicle video, detects roadway issues (RDD2022), aligns detections to GPS, and visualizes results in Rerun with a map, camera feed, and detection timeline.

### Project Layout

- `vision/` - core CV pipeline (detection, mapping, visualization)
- `webapp/` - FastAPI backend + static dashboard UI
- `webapp/static/` - HTML/CSS/JS dashboard
- `vision/demo_assets/` - demo videos, GPS CSVs, and rerun recordings
- `ARCHITECTURE.md` - deeper architecture notes + diagram

### Architecture Overview

**Core data flow**

1. **Video source** -> `vision/main.py`
2. **Detector** -> `vision/detector.py` (Ultralytics + RDD2022 label mapping)
3. **Optional visual odometry** -> `vision/vo.py` (relative motion)
4. **GPS alignment** -> `vision/gps.py` (CSV time -> lat/lon)
5. **Mapping & metrics** -> `vision/mapping.py`
6. **Visualization** -> `vision/rerun_viz.py` (Rerun map, camera, timeline)

**Webapp flow**

1. Upload video -> `webapp/app.py` (FastAPI)
2. Background job triggers pipeline command
3. Status/results served via `/status/{job_id}` and `/results/{job_id}`
4. Dashboard UI in `webapp/static/`

**Rerun layout**

- Left: map with vehicle path and geolocated detections
- Right: camera feed with color-coded boxes
- Bottom: detection timeline (counts over time)
- Selection panel: click any detection point to see metadata + cropped image

### Module Responsibilities

**`vision/main.py`**
- CLI entry point and orchestration.
- Opens video, reads frames, sets Rerun time.
- Runs detection, VO (optional), GPS lookup, mapping.
- Logs all visuals and metrics to Rerun.

**`vision/detector.py`**
- Ultralytics detector wrapper.
- Converts model labels to meaningful categories.
- Default mapping:
  - `D00` -> `longitudinal_crack`
  - `D10` -> `transverse_crack`
  - `D20` -> `alligator_crack`
  - `D40` -> `pothole`

**`vision/gps.py`**
- Loads `time_s,lat,lon,speed_mps,heading_deg`.
- Provides nearest-sample lookup for a given video timestamp.

**`vision/mapping.py`**
- Occupancy-style grid for local world-frame detections.
- Projects detections into world space using pose and camera geometry.

**`vision/vo.py`**
- Lightweight visual odometry for relative motion.
- Provides pose updates when GPS is unavailable or as a supplement.

**`vision/rerun_viz.py`**
- Rerun logging + blueprint layout.
- Map view using `GeoPoints` + `GeoLineStrings`.
- Per-detection map points with:
  - Color: red for potholes, light for cracks
  - Size: potholes slightly larger
  - Metadata: label, score, cropped image
- Detection timeline via `Scalars` under `metrics/detections/...`.

**`webapp/app.py`**
- FastAPI server for uploads, job status, and recent detection results.
- Serves the dashboard UI from `webapp/static/`.
- Runs the processing pipeline in a background task.

### Data Model

**Detections (`Detection`)**
- `label`: raw model class label
- `score`: confidence
- `bbox`: `[xmin, ymin, xmax, ymax]`
- `category`: normalized label used across the pipeline

**Pose (`Pose2D`)**
- `x, y`: world position in meters
- `yaw`: heading in radians

### Visualization Strategy

**Map overlay**
- Each detection becomes its own entity so it is selectable in Rerun.
- Hover/click to view text and cropped image in the Selection panel.

**Timeline**
- `metrics/detections/total`: total detections per frame
- `metrics/detections/by_label/...`: per-class counts
- Clicking events in the Time panel jumps the global time cursor.

### Key Files

- `webapp/app.py` - FastAPI server + static dashboard
- `webapp/static/` - dashboard UI assets
- `vision/main.py` - pipeline entry point
- `vision/detector.py` - detection + label mapping
- `vision/gps.py` - CSV GPS alignment
- `vision/rerun_viz.py` - visualization + Rerun blueprint
- `vision/mapping.py` - local mapping grid
- `vision/vo.py` - visual odometry
