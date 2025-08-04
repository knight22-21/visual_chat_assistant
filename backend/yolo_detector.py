from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or a custom-trained model later

def detect_events(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    events = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 10 == 0:  # Process every 10th frame
            results = model.predict(source=frame, conf=0.5, save=False)
            boxes = results[0].boxes.data.tolist()  # [[x1, y1, x2, y2, conf, class_id], ...]

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box[:6]
                label = model.names[int(cls_id)]

                events.append({
                    "frame": frame_id,
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)]
                })

        frame_id += 1

    cap.release()
    return events
