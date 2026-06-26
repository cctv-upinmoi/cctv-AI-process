import multiprocessing as mp
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = SCRIPT_DIR / "cctv-test.mp4"
MODEL_PATH = SCRIPT_DIR.parent / "training" / "ppe_smart_cctv" / "weights" / "best.pt"

MAX_FRAMES = 60
N_ITERS    = 30
WARMUP     = 3
CONFIGS    = [1, 4, 6, 8, 10, 20]


def load_frames(path: Path, max_frames: int):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"Không đọc được frame từ {path}")
    return frames


def worker(barrier, queue, n_iters: int):
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    frames = load_frames(VIDEO_PATH, MAX_FRAMES)

    for f in frames[:WARMUP]:
        model.track(f, persist=True, verbose=False)

    barrier.wait()                              # đồng bộ start để mọi worker đua CPU cùng lúc
    t0 = time.perf_counter()
    for i in range(n_iters):
        model.track(frames[i % len(frames)], persist=True, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) / n_iters * 1000
    queue.put(elapsed_ms)


def bench(n_workers: int) -> list[float]:
    barrier = mp.Barrier(n_workers)
    q       = mp.Queue()
    procs   = [mp.Process(target=worker, args=(barrier, q, N_ITERS)) for _ in range(n_workers)]
    for p in procs: p.start()
    for p in procs: p.join()
    return sorted(q.get() for _ in range(n_workers))


def main():
    if not VIDEO_PATH.exists():
        raise SystemExit(f"Missing video: {VIDEO_PATH}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model: {MODEL_PATH}")

    print(f"Video : {VIDEO_PATH.name}")
    print(f"Model : {MODEL_PATH.name}")
    print(f"Iters : {N_ITERS} per worker, {WARMUP} warmup")
    print()
    print(f"{'N':>4} | {'avg':>9} | {'min':>9} | {'max':>9} | {'fps/cam':>8} | {'fps tổng':>9}")
    print("-" * 64)

    for n in CONFIGS:
        try:
            times = bench(n)
            avg = sum(times) / len(times)
            print(f"{n:>4} | {avg:>7.1f}ms | {times[0]:>7.1f}ms | {times[-1]:>7.1f}ms | "
                  f"{1000/avg:>7.1f} | {n * 1000/avg:>8.1f}")
        except Exception as e:
            print(f"{n:>4} | ERROR: {e}")


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     main()
