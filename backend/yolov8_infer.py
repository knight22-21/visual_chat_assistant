from ultralytics import YOLO
from pathlib import Path
import cv2
import json

model = YOLO("yolov8n.pt")  # Use smallest model, adjust if needed

def run_yolo_on_frames(frames_dir, output_json_path):
    frames_dir = Path(frames_dir)
    results_data = {}

    for frame_path in sorted(frames_dir.glob("*.jpg")):
        result = model(frame_path, verbose=False)[0]

        frame_data = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            frame_data.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(v, 2) for v in xyxy]
            })

        results_data[frame_path.name] = frame_data

    with open(output_json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    return output_json_path
