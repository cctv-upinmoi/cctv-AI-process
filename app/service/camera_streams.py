import threading
import time

import cv2

from app.config.config import GO2RTC_RTSP_URL


class CameraStream(threading.Thread):
    def __init__(self, camera_name: str, buffer):
        super().__init__()
        self.camera_name = camera_name
        self.buffer = buffer
        self.stop_event = threading.Event()
        self._url = f"{GO2RTC_RTSP_URL}/{camera_name}"

    def run(self):
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"[{self.camera_name}] cannot open RTSP stream, retrying in 3s...")
                time.sleep(3)
                continue

            print(f"[{self.camera_name}] stream connected: {self._url}")
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.camera_name}] stream lost, reconnecting...")
                    break
                self.buffer.put(frame)

            cap.release()

    def stop(self):
        self.stop_event.set()