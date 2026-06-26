import requests

from auth.keycloak import KeycloakTokenClient
from config.config import (
    BACKEND_API_URL,
    KEYCLOAK_URL,
    KEYCLOAK_REALM,
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_CLIENT_SECRET,
)
from model.camera import Camera

_token_client = KeycloakTokenClient(
    KEYCLOAK_URL, KEYCLOAK_REALM, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET
)


def fetch_cameras() -> list[Camera]:
    try:
        token = _token_client.get_token()
        response = requests.get(
            BACKEND_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        response.raise_for_status()
        raw_cameras = response.json().get("data") or []
        return [Camera(c) for c in raw_cameras]
    except Exception as e:
        print(f"[api_client] error fetching cameras: {e}")
        return []