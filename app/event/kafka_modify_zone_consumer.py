import json
import logging
import threading

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config import config
from event.base import CameraSubscriberBase
from service.camera_config_manager import CameraConfigManager

logger = logging.getLogger(__name__)


class KafkaModifyZoneConsumer(CameraSubscriberBase):

    def __init__(self, camera_manager: CameraConfigManager):
        self.camera_manager = camera_manager
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="kafka-zone-consumer")
        self._thread.start()
        logger.info("[Kafka ZoneConsumer] Started, topic='%s'", config.KAFKA_TOPIC_MODIFY_ZONE)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        consumer = KafkaConsumer(
            config.KAFKA_TOPIC_MODIFY_ZONE,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            group_id="cctv-ai",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        logger.info("[Kafka ZoneConsumer] Connected to topic='%s'", config.KAFKA_TOPIC_MODIFY_ZONE)
        try:
            while not self._stop_event.is_set():
                records = consumer.poll(timeout_ms=1000)
                for _, messages in records.items():
                    for msg in messages:
                        self._handle(msg.value)
        except KafkaError as exc:
            logger.error("[Kafka ZoneConsumer] Error: %s", exc)
        finally:
            consumer.close()

    def _handle(self, data: dict):
        try:
            camera_id  = data["cameraId"]
            zones_data = data["zones"]
            self.camera_manager.update_camera_zones(camera_id, zones_data)
            logger.debug("[Kafka ZoneConsumer] Updated zones for cameraId=%s", camera_id)
        except KeyError as exc:
            logger.error("[Kafka ZoneConsumer] Missing field: %s", exc)
        except Exception as exc:
            logger.error("[Kafka ZoneConsumer] Unexpected error: %s", exc)
