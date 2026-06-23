import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from event.factory import make_camera_subscriber
from service.camera_config_manager import CameraConfigManager
from service.camera_streams import CameraStream
from service.frame_buffer import FrameBuffer
from service.api_client import fetch_cameras
from service.processing import AIProcessor

DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540


def _placeholder(text: str) -> np.ndarray:
    img = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, text, (DISPLAY_WIDTH // 2 - 180, DISPLAY_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
    return img


def main():
    print("Fetching cameras from backend...")
    cameras = fetch_cameras()

    active = [c for c in cameras if c.status == "OK"]
    if not active:
        print("No active cameras. Exiting.")
        return

    camera_manager = CameraConfigManager()
    camera_manager.load_initial(active)

    subscriber = make_camera_subscriber(camera_manager)
    subscriber.start()

    entries = []
    for cam in active:
        print(f"Starting stream: {cam.name}")
        buf       = FrameBuffer(maxsize=1)
        stream    = CameraStream(cam.name, buf)
        processor = AIProcessor(cam.id, cam.name, camera_manager)
        stream.daemon = True
        stream.start()
        entries.append((cam.name, stream, buf, processor))
        # Open window immediately with placeholder so it's visible right away
        # cv2.imshow(cam.name, _placeholder(f"Connecting: {cam.name}..."))

    # cv2.waitKey(1)
    # print("Press 'q' to quit.")

    try:
        while True:
            for name, _, buf, processor in entries:
                frame = buf.get()
                if frame is None:
                    # cv2.imshow(name, _placeholder(f"Waiting: {name}"))
                    continue
                frame = processor.process(frame)
                # cv2.imshow(name, cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()
        for _, stream, _, _ in entries:
            stream.stop()
        for _, stream, _, _ in entries:
            stream.join()
        # cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
