import queue


class FrameBuffer:
    def __init__(self, maxsize=1):
        self.queue = queue.Queue(maxsize=maxsize)

    def put(self, frame):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except:
                pass
        self.queue.put(frame)

    def get(self):
        if self.queue.empty():
            return None
        return self.queue.get()