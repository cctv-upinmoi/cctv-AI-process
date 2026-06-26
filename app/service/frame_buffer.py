import queue


class FrameBuffer:
    def __init__(self, maxsize=1):
        self.queue = queue.Queue(maxsize=maxsize)

    def put(self, frame):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(frame)

    def get(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None