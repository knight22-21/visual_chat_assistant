from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import shutil
import os
import json
from pydantic import BaseModel


from yolov8_infer import run_yolo_on_frames
from movinet_infer import get_actions_from_video
from frame_extractor import get_video_duration, extract_frames
from blip_infer import caption_multiple_frames
from langchain_chat import chat_with_summary  # <-- Added for chat

BASE_DIR = Path(__file__).resolve().parent.parent
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

@app.get("/ping")
def ping():
    return {"status": "ok"}

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

    # Extract 1 frame per second
    frame_output_dir = FRAME_DIR / video_filename.split('.')[0]
    extract_frames(str(video_path), str(frame_output_dir), frame_rate=1)

    # Object detection
    yolo_output_path = frame_output_dir / "yolo_output.json"
    structured, raw = run_yolo_on_frames(str(frame_output_dir), str(yolo_output_path))

    # Action recognition
    action_results = get_actions_from_video(str(video_path))

    return JSONResponse({
        "message": "Video uploaded and processed",
        "filename": video_filename,
        "structured_events": structured,
        "action_events": action_results,
        "raw_events": raw
    })

@app.post("/run-yolo")
async def run_yolo(video_filename: str):
    frame_folder = FRAME_DIR / video_filename.split('.')[0]
    if not frame_folder.exists():
        raise HTTPException(status_code=404, detail="Frame folder not found")

    output_path = frame_folder / "yolo_output.json"
    structured, raw = run_yolo_on_frames(str(frame_folder), str(output_path))

    return {
        "message": "YOLOv8 inference complete",
        "yolo_output": str(output_path),
        "structured_events": structured,
        "raw_events": raw
    }

@app.post("/frame-captions")
async def get_captions(video_filename: str):
    frame_folder = FRAME_DIR / video_filename.split('.')[0]
    if not frame_folder.exists():
        raise HTTPException(status_code=404, detail="Frame folder not found")

    captions = caption_multiple_frames(str(frame_folder), frame_sample_rate=5)

    # Save the captions for chat usage
    caption_file = frame_folder / "captions.json"
    with open(caption_file, "w") as f:
        json.dump(captions, f, indent=2)

    return {
        "message": "Captions generated for selected frames",
        "captions": captions
    }

class ChatRequest(BaseModel):
    video_filename: str
    user_input: str

@app.post("/chat")
async def chat(request: ChatRequest):
    frame_folder = FRAME_DIR / request.video_filename.split('.')[0]
    caption_file = frame_folder / "captions.json"

    if not caption_file.exists():
        raise HTTPException(status_code=404, detail="Captions file not found")

    with open(caption_file, "r") as f:
        caption_data = json.load(f)

    summary_text = "\n".join([f"{item['frame']}: {item['caption']}" for item in caption_data])
    response = chat_with_summary(summary_text, request.user_input)

    return {"response": response}
