import os

# Root of cctv-AI-process/ — config.py is at app/config/config.py, so go up 2 levels
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/cctv-core/cameras")
GO2RTC_URL      = os.getenv("GO2RTC_URL", "http://localhost:1984")
GO2RTC_RTSP_URL = os.getenv("GO2RTC_RTSP_URL", "rtsp://localhost:8554")

# Keycloak client_credentials
KEYCLOAK_URL           = os.getenv("KEYCLOAK_URL", "http://localhost:8081")
KEYCLOAK_REALM         = os.getenv("KEYCLOAK_REALM", "smart-cctv")
KEYCLOAK_CLIENT_ID     = os.getenv("KEYCLOAK_CLIENT_ID", "cctv-core")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "YjmJ5WOX84MHgVJ2IHt2xvFwirESri1O")


SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")

# Intrusion alert timing
INTRUSION_PERSIST_SECS  = float(os.getenv("INTRUSION_PERSIST_SECONDS", "0"))
INTRUSION_COOLDOWN_SECS = float(os.getenv("INTRUSION_COOLDOWN_SECONDS", "10"))

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
KAFKA_TOPIC_INTRUSION   = os.getenv("KAFKA_TOPIC_INTRUSION", "intrusion-alerts")
KAFKA_TOPIC_MODIFY_ZONE = os.getenv("KAFKA_TOPIC_MODIFY_ZONE", "modify-zone-cctv")

# Alert toggle — set ALERT_ENABLED=false to disable sending alerts to backend
ALERT_ENABLED = os.getenv("ALERT_ENABLED", "true").lower() == "true"

# YOLO model — single model for zone detection + PPE
_DEFAULT_MODEL = os.path.join(_BASE_DIR, "training", "ppe_smart_cctv", "weights", "best.pt")
YOLO_MODEL_PATH   = os.getenv("YOLO_MODEL_PATH", _DEFAULT_MODEL)

# PPE alert timing
PPE_PERSIST_SECS  = float(os.getenv("PPE_PERSIST_SECONDS", "5"))
PPE_COOLDOWN_SECS = float(os.getenv("PPE_COOLDOWN_SECONDS", "30"))