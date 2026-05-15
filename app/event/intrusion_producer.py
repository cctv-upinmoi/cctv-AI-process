import base64
import json
import logging

import cv2
import numpy as np
import redis

from app.config import config

logger = logging.getLogger(__name__)

CHANNEL = "intrusion_alerts"


class IntrusionProducer:

    def __init__(self):
        self._client = redis.Redis(
            host=config.REDIS_HOST,
            port=int(config.REDIS_PORT),
            decode_responses=False,
        )

    def send_intrusion_alert(
        self,
        timestamp: str,
        camera_id: str,
        camera_name: str,
        zone_name: str,
        confidence: float,
        frame: np.ndarray,
        alert_type: str = "INTRUSION",
        person_count: int = 1,
    ) -> None:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = json.dumps({
            "timestamp":    timestamp,
            "camera_id":    camera_id,
            "camera_name":  camera_name,
            "zone_name":    zone_name,
            "confidence":   round(confidence, 4),
            "alert_type":   alert_type,
            "person_count": person_count,
            "image":        image_b64,
        })

        try:
            self._client.publish(CHANNEL, payload)
            print("Alert published: camera=%s zone=%s", camera_name, zone_name)
        except redis.RedisError as exc:
            print("Failed to event alert: %s", exc)

    def close(self):
        self._client.close()