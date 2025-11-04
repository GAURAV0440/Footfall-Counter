import cv2
from detector import HumanDetector
from tracker import CentroidTracker
detector = HumanDetector("yolov8n.pt")
tracker = CentroidTracker()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    objects = tracker.update(detections)

    # Draw boxes + IDs
    for ((x1, y1, x2, y2, conf), (obj_id, centroid)) in zip(detections, objects.items()):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (centroid[0]-10, centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

    cv2.imshow("Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
