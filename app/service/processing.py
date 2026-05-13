import time
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from ultralytics import YOLO
from app.event.intrusion_producer import IntrusionProducer
from service.camera_config_manager import CameraConfigManager

PERSON_CLASS   = 0
MIN_CONFIDENCE = 0.5

ZONE_CLEAR_TIMEOUT = 8.0   # seconds without detection → intrusion zone re-arms
PROX_CLEAR_TIMEOUT = 4.0   # seconds without detection → proximity zone re-arms
PROXIMITY_PIXELS   = 80    # pixel distance from zone boundary → proximity warning

COLORS = {
    "safe":      (0, 255,   0),   # green
    "proximity": (0, 165, 255),   # orange
    "intrusion": (0,   0, 255),   # red
    "zone":      (0,   0, 239),   # zone border
}


class MotionDetector:
    """Lightweight pre-filter — skips YOLO when scene is static."""

    def __init__(self, min_area: int = 1500):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._min_area = min_area

    def has_motion(self, frame: np.ndarray) -> bool:
        mask = self._bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) > self._min_area for c in contours)


class AIProcessor:
    def __init__(self, camera_id: str, camera_name: str, camera_manager: CameraConfigManager, model_path="yolov8n.pt"):
        self.camera_id      = camera_id
        self.camera_name    = camera_name
        self.model          = YOLO(model_path)
        self.camera_manager = camera_manager
        self._producer      = IntrusionProducer()
        self._motion        = MotionDetector()

        # Entry-based state for INTRUSION alerts
        self._zone_last_seen: dict[str, float] = {}
        self._zone_alerted:   dict[str, bool]  = {}

        # Entry-based state for PROXIMITY alerts (independent from intrusion)
        self._prox_last_seen: dict[str, float] = {}
        self._prox_alerted:   dict[str, bool]  = {}

        # Running person counts per zone (smoothed across frames)
        self._intrusion_count: dict[str, int] = defaultdict(int)
        self._prox_count:      dict[str, int] = defaultdict(int)

    def _get_active_zones(self):
        camera = self.camera_manager.get_camera(self.camera_id)
        if not camera:
            return []
        return [
            z for z in camera.zones
            if z.enabled and z.type == "INTRUSION" and len(z.points) >= 3
        ]

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        active_zones = self._get_active_zones()

        zone_polys = [
            (z, np.array([[int(nx * w), int(ny * h)] for nx, ny in z.points], dtype=np.int32))
            for z in active_zones
        ]

        if not self._motion.has_motion(frame):
            self._tick_states(active_zones, intrusion_zones=set(), prox_zones=set())
            # Reset counts when no motion (scene is empty)
            for zone in active_zones:
                self._intrusion_count[zone.name] = 0
                self._prox_count[zone.name] = 0
            frame = self._draw_zones(frame, zone_polys)
            return frame

        results = self.model(frame, verbose=False)[0]
        now = time.time()

        intrusion_zones: set[str] = set()
        prox_zones:      set[str] = set()

        # Count persons per zone this frame
        frame_intrusion_count: dict[str, int] = defaultdict(int)
        frame_prox_count:      dict[str, int] = defaultdict(int)

        for box in results.boxes:
            if int(box.cls[0]) != PERSON_CLASS:
                continue
            conf = float(box.conf[0])
            if conf < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = y2  # bottom-center = feet position

            status, matched = self._classify_person(cx, cy, zone_polys)

            color = COLORS.get(status, COLORS["safe"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{conf:.0%}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            if matched is None:
                continue

            zone_name = matched.name

            if status == "intrusion":
                intrusion_zones.add(zone_name)
                frame_intrusion_count[zone_name] += 1
                self._zone_last_seen[zone_name] = now
                self._prox_last_seen[zone_name] = now
                self._prox_alerted[zone_name]   = True

                if not self._zone_alerted.get(zone_name, False):
                    self._zone_alerted[zone_name] = True
                    self._send_alert(
                        zone_name, conf, (cx, cy), frame,
                        alert_type="INTRUSION",
                        person_count=frame_intrusion_count[zone_name],
                    )

            elif status == "proximity":
                prox_zones.add(zone_name)
                frame_prox_count[zone_name] += 1
                self._prox_last_seen[zone_name] = now

                if not self._prox_alerted.get(zone_name, False):
                    self._prox_alerted[zone_name] = True
                    self._send_alert(
                        zone_name, conf, (cx, cy), frame,
                        alert_type="PROXIMITY",
                        person_count=frame_prox_count[zone_name],
                    )

        # Update running counts
        for zone in active_zones:
            self._intrusion_count[zone.name] = frame_intrusion_count[zone.name]
            self._prox_count[zone.name]      = frame_prox_count[zone.name]

        # Draw zones with live person counts
        frame = self._draw_zones(frame, zone_polys)

        self._tick_states(active_zones, intrusion_zones, prox_zones)
        return frame

    def _classify_person(self, cx, cy, zone_polys) -> tuple[str, object | None]:
        """
        Returns ("intrusion", zone) if inside polygon,
                ("proximity", zone) if within PROXIMITY_PIXELS of boundary,
                ("safe", None) otherwise.
        Intrusion takes priority if multiple zones match.
        """
        best_prox_zone = None
        best_prox_dist = -float("inf")

        for zone, pts in zone_polys:
            dist = cv2.pointPolygonTest(pts, (float(cx), float(cy)), measureDist=True)
            if dist >= 0:
                return "intrusion", zone
            if dist >= -PROXIMITY_PIXELS and dist > best_prox_dist:
                best_prox_dist = dist
                best_prox_zone = zone

        if best_prox_zone:
            return "proximity", best_prox_zone
        return "safe", None

    def _tick_states(self, active_zones, intrusion_zones: set[str], prox_zones: set[str]):
        """Re-arm zones that have been clear long enough."""
        now = time.time()
        for zone in active_zones:
            name = zone.name
            if name not in intrusion_zones:
                if now - self._zone_last_seen.get(name, 0) > ZONE_CLEAR_TIMEOUT:
                    self._zone_alerted[name] = False
            if name not in prox_zones and name not in intrusion_zones:
                if now - self._prox_last_seen.get(name, 0) > PROX_CLEAR_TIMEOUT:
                    self._prox_alerted[name] = False

    def _draw_zones(self, frame: np.ndarray, zone_polys) -> np.ndarray:
        for zone, pts in zone_polys:
            cv2.polylines(frame, [pts], isClosed=True, color=COLORS["zone"], thickness=2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], COLORS["zone"])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            cx_z = int(pts[:, 0].mean())
            cy_z = int(pts[:, 1].mean())

            i_count = self._intrusion_count.get(zone.name, 0)
            p_count  = self._prox_count.get(zone.name, 0)

            # Zone name label
            cv2.putText(frame, zone.name, (cx_z, cy_z - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["zone"], 2, cv2.LINE_AA)

            # Person count label — show intrusion count in red, proximity in orange
            if i_count > 0:
                label = f"xam nhap: {i_count}"
                cv2.putText(frame, label, (cx_z, cy_z + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["intrusion"], 2, cv2.LINE_AA)
            elif p_count > 0:
                label = f"lang vang: {p_count}"
                cv2.putText(frame, label, (cx_z, cy_z + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["proximity"], 2, cv2.LINE_AA)

        return frame

    def _send_alert(self, zone_name: str, conf: float, centroid: tuple,
                    frame: np.ndarray, alert_type: str = "INTRUSION",
                    person_count: int = 1):
        timestamp = datetime.now(timezone.utc).isoformat()
        self._producer.send_intrusion_alert(
            timestamp=timestamp,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            zone_name=zone_name,
            confidence=conf,
            frame=frame,
            alert_type=alert_type,
            person_count=person_count,
        )
        print(f"[{alert_type}] camera={self.camera_name} zone={zone_name} "
              f"count={person_count} conf={conf:.0%} centroid={centroid}")
