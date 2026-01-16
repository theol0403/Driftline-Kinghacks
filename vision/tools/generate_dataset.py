import argparse
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import requests

# -----------------------------------------------------------------------------
# Math & Geo Utilities
# -----------------------------------------------------------------------------

def decode_polyline(polyline_str: str, precision: int = 5) -> List[Tuple[float, float]]:
    """
    Decodes a Google Maps encoded polyline string into a list of (lat, lon) tuples.
    Adapted from common open-source implementations.
    """
    index = 0
    lat = 0
    lng = 0
    coordinates = []
    length = len(polyline_str)
    factor = 10 ** precision

    while index < length:
        b, shift, result = 0, 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / factor, lng / factor))

    return coordinates

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns distance in meters between two lat/lon points."""
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_heading(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the initial bearing from point 1 to point 2 in degrees."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lam1, lam2 = math.radians(lon1), math.radians(lon2)

    y = math.sin(lam2 - lam1) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lam2 - lam1)
    angle = math.degrees(math.atan2(y, x))
    return (angle + 360) % 360

def interpolate_path(points: List[Tuple[float, float]], step_meters: float) -> List[Tuple[float, float]]:
    """
    Resamples the path to have points approximately `step_meters` apart.
    """
    if not points:
        return []
    
    interpolated = [points[0]]
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        
        dist = haversine_distance(p1[0], p1[1], p2[0], p2[1])
        if dist < step_meters:
            continue
            
        num_steps = int(dist / step_meters)
        lat_step = (p2[0] - p1[0]) / num_steps
        lon_step = (p2[1] - p1[1]) / num_steps
        
        for s in range(1, num_steps + 1):
            new_lat = p1[0] + s * lat_step
            new_lon = p1[1] + s * lon_step
            interpolated.append((new_lat, new_lon))
            
    interpolated.append(points[-1])
    return interpolated

# -----------------------------------------------------------------------------
# Google API Interaction
# -----------------------------------------------------------------------------

def get_route(api_key: str, origin: str, destination: str) -> List[Tuple[float, float]]:
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "mode": "driving"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    if data["status"] != "OK":
        raise RuntimeError(f"Directions API Error: {data['status']} - {data.get('error_message', '')}")
        
    # Extract polyline from the first route and leg
    overview_polyline = data["routes"][0]["overview_polyline"]["points"]
    return decode_polyline(overview_polyline)

def download_street_view(
    api_key: str, 
    lat: float, 
    lon: float, 
    heading: float, 
    path: Path, 
    size: str = "640x640", 
    fov: int = 90
) -> bool:
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{lat},{lon}",
        "heading": heading,
        "fov": fov,
        "pitch": 0,
        "key": api_key,
        "source": "outdoor" # Prefer outdoor/road imagery
    }
    
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        with open(path, "wb") as f:
            f.write(resp.content)
        return True
    return False

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Street View Dashcam Video")
    parser.add_argument("--key", required=True, help="Google Maps API Key")
    parser.add_argument("--origin", required=True, help="Start location (e.g. 'Kingston City Hall, Ontario')")
    parser.add_argument("--destination", required=True, help="End location")
    parser.add_argument("--output-dir", default="data/generated", help="Output directory")
    parser.add_argument("--step-meters", type=float, default=5.0, help="Distance between frames in meters")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--size", default="640x640", help="Image size (e.g. 640x640, 2048x2048)")
    parser.add_argument("--fov", type=int, default=90, help="Field of View (default 90). Lower = Zoomed In.")
    parser.add_argument("--clean", action="store_true", help="Clean up downloaded images after video creation")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip cost confirmation")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "raw_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching route from '{args.origin}' to '{args.destination}'...")
    raw_path = get_route(args.key, args.origin, args.destination)
    print(f"Raw path has {len(raw_path)} points.")
    
    path_points = interpolate_path(raw_path, args.step_meters)
    print(f"Interpolated path has {len(path_points)} points (~{args.step_meters}m spacing).")
    
    # Estimate Cost
    cost_estimate = len(path_points) * 0.007
    print(f"ESTIMATED COST: ${cost_estimate:.2f} USD (Street View Static API)")
    
    if not args.yes:
        confirm = input("Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        
    records = []
    
    print(f"Downloading images at {args.size}, FOV {args.fov}...")
    valid_frames = 0
    
    for i in range(len(path_points) - 1):
        lat1, lon1 = path_points[i]
        lat2, lon2 = path_points[i+1]
        
        heading = calculate_heading(lat1, lon1, lat2, lon2)
        
        img_filename = f"frame_{i:06d}.jpg"
        img_path = images_dir / img_filename
        
        # Check if exists (resume capability)
        if not img_path.exists():
            success = download_street_view(args.key, lat1, lon1, heading, img_path, size=args.size, fov=args.fov)
            if not success:
                print(f"Failed to download frame {i}")
                continue
            time.sleep(0.1) # Be nice to API limits
            
        # Verify Resolution
        if i == 0:
            check_img = cv2.imread(str(img_path))
            if check_img is not None:
                h, w = check_img.shape[:2]
                print(f" [DEBUG] Actual Image Size: {w}x{h}")
                req_w, req_h = map(int, args.size.split('x'))
                if w < req_w or h < req_h:
                    print(f" [WARNING] API returned smaller image than requested. Standard API limit is usually 640x640.")
                    print(f" [TIP] Try reducing FOV (e.g. --fov 40) to zoom in and get better road details.")

        
        # Verify image is valid (API sometimes returns gray image if no data)
        # We can do a basic file size check or just trust it for now.
        
        records.append({
            "idx": i, 
            "filename": img_filename, 
            "lat": lat1, 
            "lon": lon1, 
            "heading": heading,
            "dist_m": args.step_meters # Approximate
        })
        valid_frames += 1
        print(f"\rProgress: {i+1}/{len(path_points)-1}", end="")
        
    print("\nStitching video...")
    
    video_path = output_dir / "street_view.mp4"
    csv_path = output_dir / "street_view.csv"
    
    # Initialize Video Writer
    first_img = cv2.imread(str(images_dir / records[0]["filename"]))
    if first_img is None:
        raise RuntimeError("Could not read first image.")
        
    h, w, _ = first_img.shape
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # CSV Format compatible with src/gps.py: time_s, lat, lon, speed_mps, heading_deg
        writer.writerow(["time_s", "lat", "lon", "speed_mps", "heading_deg"])
        
        video_time = 0.0
        dt = 1.0 / args.fps
        
        # Speed calculation: distance / time = speed
        # If we step 5m per frame at 10fps, speed is 50 m/s (180 km/h) -> Too fast!
        # Reality: We want realistic playback speed.
        # If we want 50km/h (~13.8 m/s). 5m spacing means a captured frame is 0.36s apart.
        # To verify: speed = step_meters / dt. 
        # If we fix step_meters (spatial) and fps (playback), speed is implicit.
        # 5m * 10fps = 50m/s = 180km/h.
        # To get 50km/h (13.8m/s), with 5m steps, we need dt = 0.36s -> ~2.7 FPS.
        # OR we need smaller steps. Google SV has discrete steps, ~5-10m.
        # Let's write the CSV with implied speed, users can just play it back.
        
        for rec in records:
            img_p = images_dir / rec["filename"]
            frame = cv2.imread(str(img_p))
            if frame is None:
                continue
            out.write(frame)
            
            speed_mps = rec["dist_m"] * args.fps
            writer.writerow([f"{video_time:.3f}", rec["lat"], rec["lon"], f"{speed_mps:.2f}", f"{rec['heading']:.2f}"])
            video_time += dt
            
    out.release()
    print(f"Video saved to {video_path}")
    print(f"GPS data saved to {csv_path}")
    
    if args.clean:
        print("Cleaning up raw images...")
        shutil.rmtree(images_dir)

if __name__ == "__main__":
    main()
