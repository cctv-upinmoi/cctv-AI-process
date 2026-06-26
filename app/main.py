import logging
import multiprocessing as mp

from service.api_client import fetch_cameras
from worker import run_camera_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [pid=%(process)d]: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Fetching cameras from backend...")
    cameras = [c for c in fetch_cameras() if c.status == "OK"]
    if not cameras:
        logger.info("No active cameras. Exiting.")
        return

    procs = [
        mp.Process(target=run_camera_worker, args=(cam,), name=f"worker-{cam.name}")
        for cam in cameras
    ]
    for p in procs:
        p.start()
        logger.info("Spawned %s (pid=%d)", p.name, p.pid)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        logger.info("Interrupt received, terminating workers...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning("Force killing %s (pid=%d)", p.name, p.pid)
                p.kill()
        logger.info("All workers stopped.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
