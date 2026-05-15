import json
import redis
from service.camera_config_manager import CameraConfigManager

CHANNEL = "modify_zone_cctv"

class ModifyCameraSubscriber:
    def __init__(self, camera_manager: CameraConfigManager, redis_client: redis.Redis):
        self.camera_manager = camera_manager
        self.pubsub = redis_client.pubsub()
        self._thread = None

    def start(self):
        self.pubsub.subscribe(**{CHANNEL: self._on_message})
        self._thread = self.pubsub.run_in_thread(sleep_time=0.01, daemon=True)
        print(f"[CameraSubscriber] Subscribed to '{CHANNEL}'")

    def stop(self):
        if self._thread:
            self._thread.stop()
        self.pubsub.close()

    def _on_message(self, message):
        print(f'[CameraSubscriber] Received message: {message}')
        if message["type"] != "message":
            return
        try:
            data = json.loads(message["data"])
            camera_id = data["cameraId"]
            zones_data = data["zones"]

            self.camera_manager.update_camera_zones(camera_id, zones_data)

        except KeyError as e:
            print(f"[CameraSubscriber] Missing field: {e}")
        except json.JSONDecodeError as e:
            print(f"[CameraSubscriber] Invalid JSON: {e}")
        except Exception as e:
            print(f"[CameraSubscriber] Unexpected error: {e}")