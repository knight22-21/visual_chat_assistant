import cv2
import torch
from transformers import AutoModelForVideoClassification, VideoMAEImageProcessor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
model = AutoModelForVideoClassification.from_pretrained(model_name).to(device)
processor = VideoMAEImageProcessor.from_pretrained(model_name)

id2label = model.config.id2label

def extract_clip(video_path, start_sec, fps=30, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    frames = []

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frames.append(frame_resized)

    cap.release()
    return frames

def get_actions_from_video(video_path, interval=3, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    duration = int(total_frames / fps)
    cap.release()

    action_results = []

    for t in range(0, duration, interval):
        clip = extract_clip(video_path, t, fps=fps, num_frames=num_frames)
        if len(clip) < num_frames:
            continue

        try:
            inputs = processor(clip, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_id = torch.argmax(outputs.logits, dim=-1).item()
                confidence = torch.softmax(outputs.logits, dim=-1)[0, predicted_id].item()
                label = id2label[predicted_id]

            action_results.append({
                "time": f"{t} sec",
                "action": label,
                "confidence": round(confidence, 2)
            })

        except Exception as e:
            print(f"Error processing clip at {t}s: {e}")

    return action_results
