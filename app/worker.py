import logging
import time

from event.factory import make_camera_subscriber
from service.camera_config_manager import CameraConfigManager
from service.camera_streams import CameraStream
from service.frame_buffer import FrameBuffer
from service.processing import AIProcessor


def run_camera_worker(camera):
    """Entry point của child process — chạy 1 camera duy nhất."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s [pid=%(process)d]: %(message)s",
    )
    logger = logging.getLogger(f"worker.{camera.name}")
    logger.info("Worker started for camera id=%s name=%s", camera.id, camera.name)

    mgr = CameraConfigManager()
    mgr.load_initial([camera])

    subscriber = make_camera_subscriber(
        mgr,
        group_id=f"cctv-ai-{camera.id}",
        camera_id_filter=str(camera.id),
    )
    subscriber.start()

    buf       = FrameBuffer(maxsize=1)
    stream    = CameraStream(camera.name, buf)
    processor = AIProcessor(camera.id, camera.name, mgr)
    stream.daemon = True
    stream.start()

    try:
        while True:
            frame = buf.get()
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                processor.process(frame)
            except Exception:
                logger.exception("error processing frame, skipping")
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Worker shutting down")
        subscriber.stop()
        stream.stop()
        stream.join(timeout=5)
        logger.info("Worker stopped")
