from abc import ABC, abstractmethod
import numpy as np


class IntrusionProducerBase(ABC):
    @abstractmethod
    def send_intrusion_alert(
        self,
        timestamp: str,
        camera_id: str,
        camera_name: str,
        zone_name: str,
        confidence: float,
        frame: np.ndarray,
        alert_type: str = "INTRUSION",
        person_count: int = 1,
        event_id: str = "",
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class CameraSubscriberBase(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...
