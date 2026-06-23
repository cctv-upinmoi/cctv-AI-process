from ultralytics import YOLO


class ModelRegistry:
    """Singleton cache — loads each YOLO model file once, reuses across detectors."""

    _models: dict[str, YOLO] = {}

    @classmethod
    def get(cls, path: str) -> YOLO:
        if path not in cls._models:
            cls._models[path] = YOLO(path)
        return cls._models[path]
