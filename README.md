## Driftline Kingston Vision

Driftline is Kingston's mobile data collection tool for public-works
intelligence. It crowdsources community-submitted dashcam footage through a
webapp and turns it into a living map of hazards and maintenance needs, so the
city can prioritize repairs faster and more fairly.

SLIDES AVAILABLE HERE: [KingHacks 2026.pdf](KingHacks%202026.pdf)

### Problem

Kingstonians run into infrastructure hazards daily: potholes, icy patches,
debris, and urban blight. Manual reporting is fragmented and slow, while city
crews need timely, geolocated insight to allocate resources effectively.

### Solution

Crowdsource community dashcam uploads through a webapp. Driftline processes each
submission, detects hazards, aligns them to GPS, and visualizes the results on a
map with a time-synced camera feed and detection timeline.

### What We Detect

- Roadway hazards: potholes, pavement cracks, sinkholes, faded lane markings
- Seasonal risks: icy patches, snow drifts, flooded sections
- Debris & obstructions: fallen branches, accident debris, illegal dump piles
- Urban blight: graffiti, trash buildup, bylaw infractions

### Architecture Overview

**Core data flow**

1. **Video source** -> `vision/main.py`
2. **Detector** -> `vision/detector.py` (Ultralytics + RDD2022 label mapping)
3. **Optional visual odometry** -> `vision/vo.py` (relative motion)
4. **GPS alignment** -> `vision/gps.py` (CSV time -> lat/lon)
5. **Mapping & metrics** -> `vision/mapping.py`
6. **Visualization** -> `vision/rerun_viz.py` (Rerun map, camera, timeline)

**Rerun layout**

- Left: map with vehicle path and geolocated detections
- Right: camera feed with color-coded boxes
- Bottom: detection timeline (counts over time)
- Selection panel: click any detection point to see metadata + cropped image
