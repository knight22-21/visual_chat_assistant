import cv2
import os

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        return 0
    return total_frames / fps

def extract_frames(video_path, output_dir, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)
    count = 0
    frame_id = 0

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
        count += 1

    cap.release()
