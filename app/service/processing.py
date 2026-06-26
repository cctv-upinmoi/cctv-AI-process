import time
import uuid
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from ultralytics import YOLO
from event.factory import make_intrusion_producer
from config.config import (
    ALERT_ENABLED,
    INTRUSION_PERSIST_SECS, INTRUSION_COOLDOWN_SECS,
    PPE_PERSIST_SECS, PPE_COOLDOWN_SECS,
    YOLO_MODEL_PATH,
)
from service.camera_config_manager import CameraConfigManager

# Class IDs trong model PPE 5-class hiện tại:
#   0: Hardhat  1: NO-Hardhat  2: NO-Safety Vest  3: Person  4: Safety Vest
PERSON_CLASS = 3  # Person

# PPE violation classes → alert type mapping
PPE_ALERT_TYPES = {
    1: "NO_HARDHAT",
    2: "NO_SAFETY_VEST",
}

PPE_CLASS_LABELS = {
    1: "NO-Hardhat",
    2: "NO-Safety Vest",
}

MIN_CONFIDENCE = 0.5

COLORS = {
    "safe":      (0, 255,   0),
    "intrusion": (0,   0, 255),
    "zone":      (0,   0, 239),
    "ppe":       (0,   0, 255),
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


class AlertGate:
    """Chống spam alert theo từng key, dùng 2 đồng hồ:
    - persist_secs : điều kiện phải đúng liên tục bao lâu mới được bắn (0 = bắn ngay).
    - cooldown_secs: bắn xong im lặng bao lâu mới được bắn lại.
    Dùng reset(key) khi điều kiện hết đúng, sweep(now) để dọn key cũ.
    """

    def __init__(self, persist_secs: float = 0.0, cooldown_secs: float = 10.0,
                 stale_secs: float = 60.0):
        self.persist  = persist_secs
        self.cooldown = cooldown_secs
        self.stale    = stale_secs
        self._first_seen: dict = {}   # key → lúc điều kiện bắt đầu đúng liên tục
        self._last_alert: dict = {}   # key → lần bắn gần nhất

    def should_fire(self, key, now: float) -> bool:
        first = self._first_seen.setdefault(key, now)
        if now - first >= self.persist and now - self._last_alert.get(key, 0.0) >= self.cooldown:
            self._last_alert[key] = now
            return True
        return False

    def reset(self, key) -> None:
        self._first_seen.pop(key, None)

    def sweep(self, now: float) -> None:
        cut = now - self.stale
        self._first_seen = {k: v for k, v in self._first_seen.items() if v > cut}
        self._last_alert = {k: v for k, v in self._last_alert.items() if v > cut}


class AIProcessor:
    def __init__(self, camera_id: str, camera_name: str, camera_manager: CameraConfigManager):
        self.camera_id      = camera_id
        self.camera_name    = camera_name
        self.model          = YOLO(YOLO_MODEL_PATH)
        self.camera_manager = camera_manager
        self._producer      = make_intrusion_producer()
        self._motion        = MotionDetector()
        self._intrusion_gate = AlertGate(persist_secs=INTRUSION_PERSIST_SECS,
                                         cooldown_secs=INTRUSION_COOLDOWN_SECS)
        self._ppe_gate       = AlertGate(persist_secs=PPE_PERSIST_SECS,
                                         cooldown_secs=PPE_COOLDOWN_SECS)
        self._intrusion_count: dict[str, int] = defaultdict(int)

        print(f"[AIProcessor] Loaded model: {YOLO_MODEL_PATH}")

    def _get_active_zones(self):
        camera = self.camera_manager.get_camera(self.camera_id)
        if not camera:
            return []
        return [z for z in camera.zones if z.enabled and len(z.points) >= 3]

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        active_zones = self._get_active_zones()

        zone_polys = [
            (z, np.array([[int(nx * w), int(ny * h)] for nx, ny in z.points], dtype=np.int32))
            for z in active_zones
        ]

        if not self._motion.has_motion(frame):
            for zone in active_zones:
                self._intrusion_count[zone.name] = 0
            for cls in PPE_ALERT_TYPES:
                self._ppe_gate.reset(cls)
            frame = self._draw_zones(frame, zone_polys)
            return frame

        # Frame sạch để build ảnh alert — tách khỏi display frame có overlay.
        raw_frame = frame.copy()

        # Single inference for zone detection + all PPE classes
        results = self.model.track(frame, persist=True, verbose=False)[0]
        now = time.time()

        frame_intrusion_count: dict[str, int] = defaultdict(int)

        # Gom PPE box theo class để build ảnh alert riêng cho từng loại vi phạm.
        ppe_boxes_by_cls: dict[int, list[tuple[int, int, int, int, float]]] = defaultdict(list)

        boxes = results.boxes
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, track_ids):
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            if conf < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            #  Person zone intrusion detection
            if cls_id == PERSON_CLASS:
                cx = (x1 + x2) // 2
                cy = y2

                in_zone, matched = self._classify_person(cx, cy, zone_polys)

                color = COLORS["intrusion"] if in_zone else COLORS["safe"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_label = f"#{track_id} " if track_id is not None else ""
                cv2.putText(frame, f"{id_label}{conf:.0%}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, color, -1)

                if not in_zone or matched is None:
                    continue

                zone_name = matched.name
                frame_intrusion_count[zone_name] += 1
                self._intrusion_count[zone_name] = frame_intrusion_count[zone_name]

                gate_key = (zone_name, track_id) if track_id is not None else (zone_name, "untracked")
                if self._intrusion_gate.should_fire(gate_key, now):
                    alert_img = self._build_intrusion_image(
                        raw_frame, (x1, y1, x2, y2), track_id, conf, zone_polys,
                    )
                    self._send_alert(
                        zone_name, conf, alert_img,
                        alert_type="INTRUSION",
                        person_count=frame_intrusion_count[zone_name],
                    )

            # PPE violation class
            elif cls_id in PPE_ALERT_TYPES:
                ppe_boxes_by_cls[cls_id].append((x1, y1, x2, y2, conf))
                label = PPE_CLASS_LABELS[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["ppe"], 2)
                cv2.putText(frame, f"{label} {conf:.0%}", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["ppe"], 1, cv2.LINE_AA)

        # Update zone counters — đảm bảo zone không có người được reset về 0.
        for zone in active_zones:
            self._intrusion_count[zone.name] = frame_intrusion_count[zone.name]

        # PPE alerts — one per class, independent cooldown.
        # Mỗi alert có ảnh CHỈ chứa bounding box của loại vi phạm đó.
        for cls_id, alert_type in PPE_ALERT_TYPES.items():
            cls_boxes = ppe_boxes_by_cls.get(cls_id, [])
            if cls_boxes:
                max_conf = max(b[4] for b in cls_boxes)
                count = len(cls_boxes)
                if self._ppe_gate.should_fire(cls_id, now):
                    alert_img = self._build_ppe_image(raw_frame, cls_id, cls_boxes)
                    self._send_alert(
                        zone_name="",
                        conf=max_conf,
                        frame=alert_img,
                        alert_type=alert_type,
                        person_count=count,
                    )
            else:
                self._ppe_gate.reset(cls_id)  # reset debounce for this class

        frame = self._draw_zones(frame, zone_polys)

        self._intrusion_gate.sweep(now)

        return frame

    def _classify_person(self, cx, cy, zone_polys) -> tuple[bool, object | None]:
        for zone, pts in zone_polys:
            dist = cv2.pointPolygonTest(pts, (float(cx), float(cy)), measureDist=False)
            if dist >= 0:
                return True, zone
        return False, None

    def _draw_zones(self, frame: np.ndarray, zone_polys) -> np.ndarray:
        for zone, pts in zone_polys:
            cv2.polylines(frame, [pts], isClosed=True, color=COLORS["zone"], thickness=2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], COLORS["zone"])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            cx_z = int(pts[:, 0].mean())
            cy_z = int(pts[:, 1].mean())

            cv2.putText(frame, zone.name, (cx_z, cy_z - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["zone"], 2, cv2.LINE_AA)

            i_count = self._intrusion_count.get(zone.name, 0)
            if i_count > 0:
                cv2.putText(frame, f"xam nhap: {i_count}", (cx_z, cy_z + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["intrusion"], 2, cv2.LINE_AA)

        return frame

    def _build_intrusion_image(self, raw_frame: np.ndarray,
                               person_box: tuple[int, int, int, int],
                               track_id, conf: float, zone_polys) -> np.ndarray:
        """Ảnh INTRUSION: zone polygon + bounding box của người vi phạm."""
        img = raw_frame.copy()
        img = self._draw_zones(img, zone_polys)
        x1, y1, x2, y2 = person_box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["intrusion"], 2)
        id_label = f"#{track_id} " if track_id is not None else ""
        cv2.putText(img, f"{id_label}{conf:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["intrusion"], 1, cv2.LINE_AA)
        return img

    def _build_ppe_image(self, raw_frame: np.ndarray, cls_id: int,
                         cls_boxes: list[tuple[int, int, int, int, float]]) -> np.ndarray:
        """Ảnh PPE: CHỈ bounding box của loại vi phạm này, không zone, không person box."""
        img = raw_frame.copy()
        label = PPE_CLASS_LABELS[cls_id]
        for x1, y1, x2, y2, conf in cls_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["ppe"], 2)
            cv2.putText(img, f"{label} {conf:.0%}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["ppe"], 1, cv2.LINE_AA)
        return img

    def _send_alert(self, zone_name: str, conf: float, frame: np.ndarray,
                    alert_type: str = "INTRUSION", person_count: int = 1):
        if not ALERT_ENABLED:
            print(f"[{alert_type}] ALERT_ENABLED=false, skipped camera={self.camera_name}")
            return
        event_id  = str(uuid.uuid4())
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
            event_id=event_id,
        )
        print(f"[{alert_type}] camera={self.camera_name} zone={zone_name} "
              f"conf={conf:.0%} event_id={event_id}")
