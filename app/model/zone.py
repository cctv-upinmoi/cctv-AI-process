class Zone:
    def __init__(self, data: dict):
        self.name = data.get("name")
        self.type = data.get("type")
        self.enabled = data.get("enabled", True)
        self.points = data.get("points", [])