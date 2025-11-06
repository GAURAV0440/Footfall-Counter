## Footfall Counter using Computer Vision

# Objective
This project is built to count how many people enter and exit through a specific area (like a door, corridor, or hallway) using computer vision.
It combines human detection, object tracking and counting logic into one clean, modular Python system.

## What I Did:
I developed a real-time footfall counter using:
YOLOv8 for detecting humans in video frames.
Centroid tracking to follow individuals frame-by-frame.
A virtual line drawn on the video feed to detect when someone crosses it.
A simple logic to count entries and exits in real time.
Output videos automatically saved with timestamps.
The system works directly through the webcam, so as soon as it runs, it starts detecting and counting people live.

## Tools and Libraries Used
Python – main programming language
OpenCV – for reading and displaying video
YOLOv8 (Ultralytics) – for human detection
NumPy & SciPy – used for tracking and centroid distance calculations
FilterPy – for smoothing object movement during tracking

## Folder Structure
<img width="358" height="438" alt="image" src="https://github.com/user-attachments/assets/fdd2bf0d-cb3f-4fa0-b10c-86700dfa61ab" />

## What Each File Does:
# main.py	The main file — opens webcam, runs detection, tracking, and counting, and saves results.
# detector.py	Handles YOLOv8-based human detection logic.
# tracker.py	Tracks detected people across frames and assigns unique IDs.
# counter.py	Checks when tracked IDs cross the ROI line and updates IN/OUT counts.
# test_detector.py	Used to test YOLO detection independently.
# test_tracker.py	Used to verify the tracking system.

## How to Run
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main program
python src/main.py

It will automatically start your webcam, detect people, and show live IN/OUT counts.
Press Q or close the window to stop the program.

## Output
Real-time video feed with bounding boxes and ID tags
IN and OUT counts displayed at the top-left
Output videos are automatically saved inside the /output/ folder
(e.g., output_webcam_20251106_231636.mp4)

## Why I Removed the “Browse Video” Option
Initially, I had added a GUI option to browse and upload videos using PyQt5.
However, during testing on Ubuntu (Wayland), it caused freezing issues inside VS Code due to Qt conflicts.
So, I decided to simplify it — now the project runs smoothly and reliably using just the webcam.

## What I Learned:

Integrating AI models (YOLOv8) into real-world video applications
Writing modular and reusable code (separate logic for detection, tracking, counting)
Implementing centroid tracking for multi-person movement
Working with OpenCV for real-time video handling
Handling cross-platform stability issues (especially on Linux Wayland).....

## THANK YOU
