import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import requests

# -----------------------------------------------------------------------------
# Mapillary Utilities
# -----------------------------------------------------------------------------

def search_sequences(token: str, bbox: List[float], min_images: int = 10) -> List[Dict[str, Any]]:
    """
    Search for sequences in a bbox (min_lon, min_lat, max_lon, max_lat).
    Returns list of sequence features.
    """
    print("Searching Mapillary API (via images endpoint)...")
    
    # API v4: /images with bbox
    url = "https://graph.mapillary.com/images"
    headers = {
        "Authorization": f"OAuth {token}"
    }
    params = {
        "fields": "id,sequence,captured_at,geometry,compass_angle",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": 500  # Fetch enough images to find sequences
    }
    
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"[API ERROR] Status: {resp.status_code}")
        print(f"[API ERROR] Body: {resp.text}")
        return []
        
    data = resp.json()
    images = data.get("data", [])
    print(f"Found {len(images)} images in search area.")
    
    # Group by sequence
    sequences = {}
    for img in images:
        seq_id = img.get("sequence")
        if not seq_id:
            continue
            
        if seq_id not in sequences:
            sequences[seq_id] = []
        sequences[seq_id].append(img)
    
    print(f"Grouped into {len(sequences)} unique sequences.")
    
    results = []
    for seq_id, imgs in sequences.items():
        if len(imgs) < min_images:
            continue
            
        # Sort by capture time
        imgs.sort(key=lambda x: x.get("captured_at", ""))
        
        # Get location of first image
        coords = imgs[0]["geometry"]["coordinates"]
        
        results.append({
            "id": seq_id,
            "count": len(imgs),
            "start_time": imgs[0].get("captured_at"),
            "start_loc": coords,
            "images": imgs
        })
        
    return results

def get_image_ids_for_sequence(token: str, sequence_id: str) -> List[Dict[str, Any]]:
    """Fetches all image IDs for a sequence."""
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "sequence_id": sequence_id,
        "fields": "id,geometry,compass_angle,captured_at",
        "limit": 2000
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("data", [])
    print(f"Error fetching sequence images: {resp.text}")
    return []

def get_image_url(token: str, image_id: str) -> str:
    """Gets the download URL for a specific image ID (high res)."""
    url = f"https://graph.mapillary.com/{image_id}"
    params = {
        "access_token": token,
        "fields": "thumb_original_url" 
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("thumb_original_url")
    return None

def download_image(url: str, path: Path) -> bool:
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Mapillary Dashcam Video")
    parser.add_argument("--token", required=True, help="Mapillary Client Access Token")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=('min_lon', 'min_lat', 'max_lon', 'max_lat'), 
                        help="Bounding box to search for sequences")
    parser.add_argument("--sequence-id", help="Directly download a specific sequence ID")
    parser.add_argument("--output-dir", default="data/mapillary", help="Output directory")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--clean", action="store_true", help="Clean raw images after stitching")
    parser.add_argument("--limit-frames", type=int, default=0, help="Limit number of frames to download (0 = all)")
    parser.add_argument("--min-frames", type=int, default=50, help="Minimum number of frames for a sequence to be listed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "raw_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    sequence_to_process = None
    
    if args.sequence_id:
        # If ID provided, we need to fetch its images first to get metadata
        # Re-using search logic but just one ID would be cleaner, but for now 
        # let's assume we search or just query images for that sequence.
        # simpler: Query images by sequence_id is not a direct filter in some endpoints, 
        # but let's try the sequence node.
        pass # TODO: Implement direct ID fetch if needed, but searching is safer for now.
        print("Note: Direct sequence ID fetch requires image list query. For now please use --bbox to discover.")
        
    if args.bbox:
        # mly_interface.set_access_token(args.token) # Not needed for requests
        seqs = search_sequences(args.token, args.bbox, min_images=args.min_frames)
        
        if not seqs:
            print("No suitable sequences found.")
            sys.exit(0)
            
        print("\nAvailable Sequences (Top 20):")
        # Sort by time just to have order
        seqs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        
        for idx, s in enumerate(seqs[:20]):
            print(f" [{idx}] ID: {s['id']} | Frames: {s['count']} | Start: {s['start_time']} | Loc: {s['start_loc']}")
            
        sel = input("\nSelect sequence index to download (or q to quit): ")
        if sel.lower() == 'q':
            sys.exit(0)
            
        try:
            sequence_to_process = seqs[int(sel)]
        except (ValueError, IndexError):
            print("Invalid selection.")
            sys.exit(1)
            
    if not sequence_to_process:
        print("Please provide --bbox to populate sequences.")
        sys.exit(1)
        
    # Process
    print(f"Processing Sequence {sequence_to_process['id']} ({sequence_to_process['count']} frames)...")
    images = sequence_to_process['images']
    
    if args.limit_frames > 0:
        images = images[:args.limit_frames]
        
    records = []
    
    for i, img in enumerate(images):
        img_id = img["id"]
        img_filename = f"frame_{i:06d}.jpg"
        img_path = images_dir / img_filename
        
        # Metadata
        coords = img["geometry"]["coordinates"]
        heading = img.get("compass_angle", 0)
        captured_at = img.get("captured_at") # Timestamp string
        
        # Download
        if not img_path.exists():
            # Get URL
            dl_url = get_image_url(args.token, img_id)
            if not dl_url:
                print(f"Failed to get URL for image {img_id}")
                continue
                
            print(f"\rDownloading {i+1}/{len(images)}: {img_id}", end="")
            success = download_image(dl_url, img_path)
            if not success:
                print(f" - Failed download")
                continue
            time.sleep(0.05)
            
        records.append({
            "filename": img_filename,
            "lat": coords[1],
            "lon": coords[0],
            "heading": heading,
            "time_s": i * (1/args.fps) # Mock time if we don't parse ISO string perfectly
        })
        
    print("\nStitching video...")
    video_path = output_dir / "mapillary_video.mp4"
    csv_path = output_dir / "mapillary.csv"
    
    if not records:
        print("No records to stitch.")
        sys.exit(1)

    first_img = cv2.imread(str(images_dir / records[0]["filename"]))
    if first_img is None:
        raise RuntimeError("Could not read first image.")
    h, w = first_img.shape[:2]
    
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "lat", "lon", "speed_mps", "heading_deg"])
        
        for r in records:
            frame = cv2.imread(str(images_dir / r["filename"]))
            if frame is None:
                continue
            # Helper: Resize if inconsistent? Mapillary sequences usually consistent.
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
                
            out.write(frame)
            writer.writerow([f"{r['time_s']:.3f}", r["lat"], r["lon"], "10.0", r["heading"]])
            
    out.release()
    print(f"Saved to {video_path}")
    print(f"GPS saved to {csv_path}")
    
    if args.clean:
        shutil.rmtree(images_dir)

if __name__ == "__main__":
    main()
