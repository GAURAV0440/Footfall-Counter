import cv2
from ultralytics import YOLO

class HumanDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.55):
        self.model = YOLO("models/yolov8n.pt")
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf >= self.conf_threshold:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

        return detections