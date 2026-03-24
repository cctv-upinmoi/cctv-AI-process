from app.camera_streams import CameraStream
from app.frame_buffer import FrameBuffer
from app.processing import AIProcessor


class App:
    def __init__(self, rtsp_url):
        self.buffer = FrameBuffer(maxsize=1)

        zone_points = [(100, 100), (500, 100), (500, 300), (100, 300)]

        self.camera = CameraStream(rtsp_url, self.buffer)
        self.ai = AIProcessor(self.buffer, zone_points)

    def start(self):
        self.camera.start()
        self.ai.start()

    def stop(self):
        self.camera.stop()
        self.ai.stop()

        self.camera.join()
        self.ai.join()


if __name__ == "__main__":
    # call restapi to core load cameras information
    RTSP_URL = "rtsp://localhost:8554/camera1"

    app = App(RTSP_URL)

    try:
        app.start()
    except KeyboardInterrupt:
        print("Stopping...")
        app.stop()