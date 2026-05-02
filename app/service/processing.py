import time
import cv2
import numpy as np
from datetime import datetime, timezone
from ultralytics import YOLO
from app.event.intrusion_producer import IntrusionProducer
from service.camera_config_manager import CameraConfigManager

PERSON_CLASS   = 0
MIN_CONFIDENCE = 0.5
ALERT_COOLDOWN = 5.0  # seconds between alerts per zone

COLORS = {
    "safe":      (0, 255, 0),    # green — person outside zone
    "intrusion": (0, 0, 255),    # red   — person inside zone
    "zone":      (0, 0, 239),    # red   — zone border
}


class AIProcessor:
    def __init__(self, camera_id: str, camera_name: str, camera_manager: CameraConfigManager, model_path="yolov8n.pt"):
        self.camera_id   = camera_id
        self.camera_name = camera_name
        self.model       = YOLO(model_path)
        self.camera_manager = camera_manager
        self._last_alert: dict[str, float] = {}
        self._producer = IntrusionProducer()

    def _get_active_zones(self):
        camera = self.camera_manager.get_camera(self.camera_id)
        if not camera:
            return []
        return [
            z for z in camera.zones
            if z.enabled and z.type == "INTRUSION" and len(z.points) >= 3
        ]

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO detection, draw results, return annotated frame."""
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]

        # get zone latest per frame
        active_zones = self._get_active_zones()

        # Pre-compute zone polygons in pixel coords once per frame
        zone_polys = [
            (z, np.array([[int(nx * w), int(ny * h)] for nx, ny in z.points], dtype=np.int32))
            for z in active_zones
        ]

        pending_alerts = []

        for box in results.boxes:
            if int(box.cls[0]) != PERSON_CLASS:
                continue
            conf = float(box.conf[0])
            if conf < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            in_zone, matched = self._check_zones(cx, cy, zone_polys)
            color = COLORS["intrusion"] if in_zone else COLORS["safe"]

            # Bounding box + confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{conf:.0%}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Centroid dot
            cv2.circle(frame, (cx, cy), 4, color, -1)

            if in_zone and matched:
                pending_alerts.append((matched.name, conf, (cx, cy)))

        # Draw zones before saving snapshots so they appear in the image
        for zone, pts in zone_polys:
            cv2.polylines(frame, [pts], isClosed=True, color=COLORS["zone"], thickness=2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], COLORS["zone"])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cx_z = int(pts[:, 0].mean())
            cy_z = int(pts[:, 1].mean())
            cv2.putText(frame, zone.name, (cx_z, cy_z),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["zone"], 2, cv2.LINE_AA)

        for zone_name, conf, centroid in pending_alerts:
            self._trigger_alert(zone_name, conf, centroid, frame)

        return frame

    def _check_zones(self, cx, cy, zone_polys):
        for zone, pts in zone_polys:
            if cv2.pointPolygonTest(pts, (cx, cy), measureDist=False) >= 0:
                return True, zone
        return False, None

    def _trigger_alert(self, zone_name: str, conf: float, centroid: tuple, frame: np.ndarray):
        now = time.time()
        if now - self._last_alert.get(zone_name, 0) < ALERT_COOLDOWN:
            return
        self._last_alert[zone_name] = now

        timestamp = datetime.now(timezone.utc).isoformat()
        self._producer.send_intrusion_alert(
            timestamp=timestamp,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            zone_name=zone_name,
            confidence=conf,
            frame=frame,
        )
        print(f"[ALERT] camera={self.camera_name} zone={zone_name} "
              f"conf={conf:.0%} centroid={centroid} ts={timestamp}")