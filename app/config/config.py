import os

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/cctv-core/cameras")
GO2RTC_URL      = os.getenv("GO2RTC_URL", "http://localhost:1984")
GO2RTC_RTSP_URL = os.getenv("GO2RTC_RTSP_URL", "rtsp://localhost:8554")

# Keycloak client_credentials
KEYCLOAK_URL           = os.getenv("KEYCLOAK_URL", "http://localhost:8081")
KEYCLOAK_REALM         = os.getenv("KEYCLOAK_REALM", "smart-cctv")
KEYCLOAK_CLIENT_ID     = os.getenv("KEYCLOAK_CLIENT_ID", "cctv-ai")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "X7Bnd7bSLTd6bb10OyZhh4mc166TY6fC")

# Redis env
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")


SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
