from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from yolov8_infer import run_yolo_on_frames
import uuid
import shutil
import os

from frame_extractor import get_video_duration, extract_frames

BASE_DIR = Path(__file__).resolve().parent.parent  # One level up from current file
UPLOAD_DIR = BASE_DIR / "videos"
FRAME_DIR = BASE_DIR / "frames"
MAX_VIDEO_DURATION = 120  # seconds

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    uid = uuid.uuid4().hex
    video_filename = f"{uid}_{video.filename}"
    video_path = UPLOAD_DIR / video_filename

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    duration = get_video_duration(str(video_path))
    if duration > MAX_VIDEO_DURATION:
        video_path.unlink()
        raise HTTPException(status_code=400, detail="Video exceeds 2-minute limit")

    frame_output_dir = FRAME_DIR / video_filename.split('.')[0]
    extract_frames(str(video_path), str(frame_output_dir), frame_rate=1)

    return {
        "message": "Video uploaded and frames extracted",
        "video_path": str(video_path),
        "frames_dir": str(frame_output_dir)
    }

@app.post("/run-yolo")
async def run_yolo(video_filename: str):
    frame_folder = FRAME_DIR / video_filename.split('.')[0]
    if not frame_folder.exists():
        raise HTTPException(status_code=404, detail="Frame folder not found")

    output_path = frame_folder / "yolo_output.json"
    result_path = run_yolo_on_frames(str(frame_folder), str(output_path))

    return {
        "message": "YOLOv8 inference complete",
        "yolo_output": str(result_path)
    }
