import threading
import time
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
# from ultralytics import YOLO

class AIProcessor(threading.Thread):
    def __init__(self, camera_id, buffer, zones_data, model_path="yolov8n.pt", skip_frame=3):
        super().__init__()
        self.camera_id = camera_id
        self.buffer = buffer
        
        self.zones = []
        if zones_data:
            for z in zones_data:
                # zones_data is a list of Zone objects
                if getattr(z, "enabled", True) and getattr(z, "points", []):
                    self.zones.append(Polygon(z.points))
        
        if not self.zones:
            # Fallback if no valid zones are provided
            self.zones.append(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

        self.model = YOLO(model_path)
        self.skip_frame = skip_frame
        self.frame_count = 0
        self.stop_event = threading.Event()
        self.alerted = set()

    def run(self):
        while not self.stop_event.is_set():
            frame = self.buffer.get()

            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            if self.frame_count % self.skip_frame != 0:
                continue

            self.process_frame(frame)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != 0 or conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            point = Point(cx, cy)
            
            in_any_zone = False
            for zone in self.zones:
                if zone.contains(point):
                    in_any_zone = True
                    break

            if in_any_zone:
                person_id = f"{self.camera_id}-{cx}-{cy}"

                if person_id not in self.alerted:
                    print(f"🚨 ALERT [{self.camera_id}]: Person detected in zone at ({cx}, {cy})")
                    self.alerted.add(person_id)

            # draw
            color = (0, 0, 255) if in_any_zone else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        for zone in self.zones:
            pts = np.array(zone.exterior.coords, np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        window_name = f"AI CCTV - {self.camera_id}"
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            self.stop()

    def stop(self):
        self.stop_event.set()
        cv2.destroyAllWindows()