import cv2
import time

class PeopleCounter:
    def __init__(self, line_position=300, min_move_threshold=25, cooldown_time=1.2):
        self.line_position = line_position
        self.min_move_threshold = min_move_threshold
        self.cooldown_time = cooldown_time
        self.total_in = 0
        self.total_out = 0
        self.memory = {}
        self.last_count_time = {}

    def update_counts(self, objects):
        current_time = time.time()

        for object_id, centroid in objects.items():
            if object_id not in self.memory:
                self.memory[object_id] = [centroid]
                self.last_count_time[object_id] = 0
            else:
                self.memory[object_id].append(centroid)

                # Compare last two centroid positions
                if len(self.memory[object_id]) >= 2:
                    prev, curr = self.memory[object_id][-2], self.memory[object_id][-1]
                    dx = curr[0] - prev[0]
                    moved_enough = abs(dx) > self.min_move_threshold
                    time_since_last = current_time - self.last_count_time.get(object_id, 0)

                    if moved_enough and time_since_last >= self.cooldown_time:
                        # IN (Left → Right)
                        if prev[0] < self.line_position <= curr[0]:
                            self.total_in += 1
                            self.last_count_time[object_id] = current_time
                        # OUT (Right → Left)
                        elif prev[0] > self.line_position >= curr[0]:
                            self.total_out += 1
                            self.last_count_time[object_id] = current_time

    def draw_info(self, frame):
        cv2.line(frame, (self.line_position, 0),
                 (self.line_position, frame.shape[0]), (255, 255, 0), 2)
        cv2.putText(frame, f"IN: {self.total_in}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.total_out}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
