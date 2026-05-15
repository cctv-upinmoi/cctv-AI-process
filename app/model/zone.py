class Zone:
    def __init__(self, data: dict):
        self.name = data.get("name")
        self.type = data.get("type")
        self.enabled = data.get("enabled", True)
        self.points = data.get("points", [])

    def __repr__(self):
        return f"Zone(name={self.name}, type={self.type}, enabled={self.enabled})"