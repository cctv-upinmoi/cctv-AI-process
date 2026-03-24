import threading
import cv2
import time


class CameraStream(threading.Thread):
    def __init__(self, rtsp_url, buffer, width=640, height=360):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.buffer = buffer
        self.width = width
        self.height = height
        self.stop_event = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)

        while not self.stop_event.is_set():
            ret, frame = cap.read()

            if not ret:
                print("⚠️ Reconnecting stream...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            frame = cv2.resize(frame, (self.width, self.height))
            self.buffer.put(frame)

        cap.release()

    def stop(self):
        self.stop_event.set()