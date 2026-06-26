import threading
from typing import Dict, Optional
from model.camera import Camera
from model.zone import Zone


class CameraConfigManager:
    def __init__(self):
        self._cameras: Dict[str, Camera] = {}
        self._lock = threading.RLock()

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        with self._lock:
            return self._cameras.get(str(camera_id))

    def get_zones(self, camera_id: str):
        camera = self.get_camera(camera_id)
        return camera.zones if camera else []

    def update_camera_zones(self, camera_id: str, zones_data: list):
        with self._lock:
            camera = self._cameras.get(str(camera_id))
            if camera is None:
                print(f"[CameraConfig] '{camera_id}' not found. Keys: {list(self._cameras.keys())}")
                return
            camera.zones = [Zone(z) for z in zones_data]
            print(f"[CameraConfig] '{camera_id}' updated → {len(camera.zones)} zones")
            for z in camera.zones:
                print(f"  {z}")

    def load_initial(self, cameras: list):
        with self._lock:
            for cam in cameras:
                self._cameras[str(cam.id)] = cam
        print(f"[CameraConfig] Loaded {len(self._cameras)} cameras")