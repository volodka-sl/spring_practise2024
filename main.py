import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.positions = [(detection["center_x"], detection["center_y"])]
        self.frames_since_seen = 0
        self.counted = False
        self.direction = None  # 'up' или 'down'
        
    def update(self, detection):
        self.positions.append((detection["center_x"], detection["center_y"]))
        self.frames_since_seen = 0
        
        if len(self.positions) >= 2 and not self.direction:
            y_diff = self.positions[-1][1] - self.positions[0][1]
            self.direction = "up" if y_diff < 0 else "down"
            
    def get_predicted_position(self):
        if len(self.positions) < 2:
            return self.positions[-1]
        dx = self.positions[-1][0] - self.positions[-2][0]
        dy = self.positions[-1][1] - self.positions[-2][1]
        return (self.positions[-1][0] + dx, self.positions[-1][1] + dy)

class CarTracker:
    def __init__(self, max_frames_to_skip=10, max_distance=100):
        self.tracks = []
        self.track_id_count = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.max_distance = max_distance
        self.counts = {"up": 0, "down": 0}
        self.counting_line_y = None
        
    def calculate_distance(self, detection, track):
        pred_pos = track.get_predicted_position()
        dx = detection["center_x"] - pred_pos[0]
        dy = detection["center_y"] - pred_pos[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def update(self, detections, frame_height):
        if self.counting_line_y is None:
            self.counting_line_y = frame_height // 2

        if not detections:
            for track in self.tracks:
                track.frames_since_seen += 1
            return

        detection_points = []
        for det in detections:
            center_x = (det[0] + det[2]) / 2
            center_y = (det[1] + det[3]) / 2
            detection_points.append({
                "center_x": center_x,
                "center_y": center_y,
                "bbox": det
            })

        used_detections = set()

        if self.tracks:
            distance_matrix = []
            for track in self.tracks:
                distances = [self.calculate_distance(det, track) for det in detection_points]
                distance_matrix.append(distances)

            for i, track in enumerate(self.tracks):
                if track.frames_since_seen > self.max_frames_to_skip:
                    continue
                    
                distances = distance_matrix[i]
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                if min_dist <= self.max_distance and min_dist_idx not in used_detections:
                    track.update(detection_points[min_dist_idx])
                    used_detections.add(min_dist_idx)
                    
                    if not track.counted:
                        last_pos = track.positions[-1][1]
                        if len(track.positions) >= 2:
                            prev_pos = track.positions[-2][1]
                            if (prev_pos < self.counting_line_y and last_pos >= self.counting_line_y) or \
                               (prev_pos > self.counting_line_y and last_pos <= self.counting_line_y):
                                track.counted = True
                                if track.direction == "up":
                                    self.counts["up"] += 1
                                else:
                                    self.counts["down"] += 1

        for i, det in enumerate(detection_points):
            if i not in used_detections:
                self.tracks.append(Track(det, self.track_id_count))
                self.track_id_count += 1

        self.tracks = [track for track in self.tracks if track.frames_since_seen <= self.max_frames_to_skip]

def process_video(video_path):
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = CarTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, classes=[2, 5, 7])
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append([x1, y1, x2, y2])
        
        tracker.update(detections, frame_height)
        
        cv2.line(frame, (0, frame_height//2), (frame_width, frame_height//2), (0, 255, 0), 2)
        
        for track in tracker.tracks:
            if track.positions:
                x, y = track.positions[-1]
                for det in detections:
                    det_center_x = (det[0] + det[2]) / 2
                    det_center_y = (det[1] + det[3]) / 2
                    if abs(det_center_x - x) < 10 and abs(det_center_y - y) < 10:
                        cv2.rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (0, 255, 0), 2)

                        id_text = f"ID: {track.track_id}"
                        text_x = int(det[0])
                        text_y = int(det[1] - 10)
                        (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                        cv2.rectangle(frame, (text_x, text_y - text_h), (text_x + text_w, text_y + 5), (255, 255, 255), -1)
                        cv2.putText(frame, id_text, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        break
            
        cv2.putText(frame, f"Cars up: {tracker.counts['up']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Cars down: {tracker.counts['down']}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Car Counting", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "output2.avi"
    process_video(video_path)
