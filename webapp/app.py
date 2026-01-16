import os
import subprocess
import threading
import uuid
import sqlite3
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DB_DIR = Path("runs")
DB_DIR.mkdir(exist_ok=True)

# Store job status in memory
jobs = {}

def run_pipeline(job_id: str, video_path: str, db_path: str):
    jobs[job_id]["status"] = "processing"
    
    # Command to run the pipeline
    # Using python -m src.pipeline to ensure imports work correctly
    cmd = [
        "python", "-m", "src.pipeline",
        "--source", video_path,
        "--model", "yolov8_pothole.pt",
        "--sqlite-path", db_path,
        "--max-frames", "300" # Limit to 300 frames for detection demo
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        jobs[job_id]["pid"] = process.pid
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            jobs[job_id]["status"] = "completed"
        else:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = stderr
            print(f"Pipeline error: {stderr}")
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"Exception running pipeline: {e}")

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    db_path = DB_DIR / f"{job_id}.sqlite"
    
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "status": "pending",
        "progress": 0,
        "db_path": str(db_path)
    }
    
    background_tasks.add_task(run_pipeline, job_id, str(video_path), str(db_path))
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    job = jobs[job_id]
    
    # Check detections count if DB exists
    detections_count = 0
    if os.path.exists(job["db_path"]):
        try:
            conn = sqlite3.connect(job["db_path"])
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM detections")
            detections_count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass
            
    return {
        "status": job["status"],
        "detections": detections_count,
        "filename": job["filename"]
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    db_path = jobs[job_id]["db_path"]
    if not os.path.exists(db_path):
        return {"results": []}
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY time_s DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    
    return {"results": [dict(row) for row in rows]}

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
