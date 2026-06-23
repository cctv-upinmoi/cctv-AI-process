import base64
import json
import logging

import cv2
import numpy as np
from kafka import KafkaProducer
from kafka.errors import KafkaError

from config import config
from event.base import IntrusionProducerBase

logger = logging.getLogger(__name__)


class KafkaIntrusionProducer(IntrusionProducerBase):

    def __init__(self):
        self._producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: v,
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
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
        event_id: str = "",
    ) -> None:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = json.dumps({
            "event_id":     event_id,
            "timestamp":    timestamp,
            "camera_id":    camera_id,
            "camera_name":  camera_name,
            "zone_name":    zone_name,
            "confidence":   round(confidence, 4),
            "alert_type":   alert_type,
            "person_count": person_count,
            "image":        image_b64,
        }).encode("utf-8")

        future = self._producer.send(
            config.KAFKA_TOPIC_INTRUSION,
            key=camera_id,
            value=payload,
        )
        try:
            future.get(timeout=5)
            logger.info("Alert published to Kafka: camera=%s zone=%s event_id=%s",
                        camera_name, zone_name, event_id)
        except KafkaError as exc:
            logger.error("Failed to publish alert to Kafka: %s", exc)

    def close(self):
        self._producer.flush()
        self._producer.close()
