import cv2

from event.modify_zone_consumer import ModifyCameraSubscriber
from service.camera_config_manager import CameraConfigManager
from service.camera_streams import CameraStream
from service.frame_buffer import FrameBuffer
from service.api_client import fetch_cameras
from service.processing import AIProcessor
import redis
from app.config import config

DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540


def main():
    print("Fetching cameras from backend...")
    cameras = fetch_cameras()

    active = [c for c in cameras if c.status == "OK"]
    if not active:
        print("No active cameras. Exiting.")
        return

    redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
    camera_manager = CameraConfigManager(redis_client)
    camera_manager.load_initial(active)

    subscriber = ModifyCameraSubscriber(camera_manager, redis_client)
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

    print("Press 'q' to quit.")
    try:
        while True:
            for name, _, buf, processor in entries:
                frame = buf.get()
                if frame is None:
                    continue
                frame = processor.process(frame)
                cv2.imshow(name, cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()
        for _, stream, _, _ in entries:
            stream.stop()
        for _, stream, _, _ in entries:
            stream.join()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()