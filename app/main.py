import time
from app.camera_streams import CameraStream
from app.frame_buffer import FrameBuffer
# from app.processing import AIProcessor
from app.api_client import fetch_cameras


class App:
    def __init__(self, camera):
        self.camera_id = camera.indexId
        self.buffer = FrameBuffer(maxsize=1)

        self.camera_stream = CameraStream(camera.rtsp_stream_url, self.buffer)
        # self.ai = AIProcessor(self.camera_id, self.buffer, camera.zones)

    def start(self):
        self.camera_stream.start()
        # self.ai.start()

    def stop(self):
        self.camera_stream.stop()
        # self.ai.stop()

    def join(self):
        self.camera_stream.join()
        # self.ai.join()


if __name__ == "__main__":
    print("Fetching cameras from backend API...")
    cameras = fetch_cameras()
    
    if not cameras:
        print("No camera data found. Exiting.")
        exit(1)

    for cam in cameras:
        print(f"[{cam.id}] {cam.name} - {cam.rtsp_stream_url}")
        for zone in cam.zones:
            print(f"  Zone: {zone.name} ({zone.type}) - enabled: {zone.enabled}")

    # apps = []
    #
    # for cam in cameras:
    #     cam_id = cam.indexId
    #     status = cam.status or "NOK"
    #
    #     if status != "OK":
    #         print(f"Skipping camera {cam_id} because status is {status}")
    #         continue
    #
    #     if cam.rtsp_stream_url:
    #         print(f"Initializing stream for camera {cam_id}: {cam.rtsp_stream_url}")
    #         # app_instance = App(cam)
    #         apps.append(app_instance)
    #     else:
    #         print(f"Skipping camera {cam_id} because RTSP Url is empty.")
    #
    # if not apps:
    #     print("No active cameras to process. Exiting.")
    #     exit(1)
    #
    # print("Starting all active cameras...")
    # try:
    #     for app_instance in apps:
    #         app_instance.start()
    #
    #     while True:
    #         time.sleep(1)
            
    # except KeyboardInterrupt:
    #     print("\nStopping...")
    #     for app_instance in apps:
    #         app_instance.stop()
    #     for app_instance in apps:
    #         app_instance.join()
    #     print("All cameras stopped.")