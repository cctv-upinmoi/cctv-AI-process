import requests

from app.model.camera import Camera
from config import BACKEND_API_URL

def fetch_cameras():
    try:
        response = requests.get(BACKEND_API_URL, timeout=10, verify=False)
        result = response.json()
        response.raise_for_status()
        # print(result)
        raw_cameras = result.get("data", [])
        return [Camera(c) for c in raw_cameras]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching cameras from backend ({BACKEND_API_URL}): {e}")
        return []
