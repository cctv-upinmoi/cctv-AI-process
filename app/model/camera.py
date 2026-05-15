from app.model.zone import Zone


class Camera:
    def __init__(self, data: dict):
        self.id = data.get("id")
        self.index_id = data.get("indexId")
        self.name = data.get("name")
        self.ip = data.get("ip")
        self.port = data.get("port")
        self.username = data.get("username")
        self.status = data.get("status")
        self.mode = data.get("mode")
        self.rtsp_stream_url = data.get("rtspStreamUrl")
        self.longitude = data.get("longitude")
        self.latitude = data.get("latitude")
        self.location_detail = data.get("locationDetail")
        self.zones = [Zone(z) for z in (data.get("zones") or [])]
        self.created_at = data.get("createdAt")
        self.updated_at = data.get("updatedAt")

    def __repr__(self):
        return f"Camera(id={self.id}, name={self.name}, status={self.status})"

