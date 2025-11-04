import os, sys, cv2, shutil
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QMessageBox
from detector import HumanDetector
from tracker import CentroidTracker
from counter import PeopleCounter

class InputSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.source = {"path": None, "name": None}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🎥 Footfall Counter")
        self.setGeometry(700, 300, 350, 200)
        layout = QVBoxLayout()

        label = QLabel("Select Input Source")
        label.setStyleSheet("font-size:16px; font-weight:bold; margin-bottom:10px;")
        layout.addWidget(label)

        webcam_btn = QPushButton("📹 Use Webcam")
        webcam_btn.setStyleSheet("background-color:#0078D7; color:white; font-size:14px; padding:8px;")
        webcam_btn.clicked.connect(self.use_webcam)
        layout.addWidget(webcam_btn)

        video_btn = QPushButton("🎞️ Browse Video File")
        video_btn.setStyleSheet("background-color:#28A745; color:white; font-size:14px; padding:8px;")
        video_btn.clicked.connect(self.browse_video)
        layout.addWidget(video_btn)

        self.setLayout(layout)

    def use_webcam(self):
        self.source["path"] = 0
        self.source["name"] = "webcam"
        self.close()

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", os.path.join(os.getcwd(), "data"),
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            os.makedirs("data", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_path = os.path.join("data", f"user_video_{timestamp}.mp4")
            shutil.copy(file_path, dest_path)
            self.source["path"] = dest_path
            self.source["name"] = os.path.basename(dest_path)
        self.close()

def main():
    app = QApplication(sys.argv)
    selector = InputSelector()
    selector.show()
    app.exec_()
    source = selector.source
    if source["path"] is None:
        sys.exit(0)

    detector = HumanDetector("models/yolov8n.pt")
    tracker = CentroidTracker(max_disappeared=30)

    cap = cv2.VideoCapture(source["path"])
    ret, frame = cap.read()
    if not ret:
        sys.exit(0)

    h, w = frame.shape[:2]
    counter = PeopleCounter(line_position=w // 2)
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"output_{source['name'].split('.')[0]}_{timestamp}.mp4"
    out = cv2.VideoWriter(os.path.join("output", output_name), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))

    print(f" Started Processing: {source['name']}\nPress 'Q' or click ❌ to quit.\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        objects = tracker.update(detections)
        counter.update_counts(objects)

        for ((x1, y1, x2, y2, _), (obj_id, c)) in zip(detections, objects.items()):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (c[0]-10, c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, c, 4, (0, 0, 255), -1)

        frame = counter.draw_info(frame)
        out.write(frame)
        cv2.imshow("Footfall Counter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        try:
            if cv2.getWindowProperty("Footfall Counter", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release(); out.release(); cv2.destroyAllWindows()
    msg = QMessageBox()
    msg.setWindowTitle(" Processing Complete")
    msg.setText(f"IN: {counter.total_in}\nOUT: {counter.total_out}\n\nSaved as {output_name}")
    msg.exec_()

if __name__ == "__main__":
    main()
