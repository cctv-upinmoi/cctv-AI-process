import time
import requests


class KeycloakTokenClient:
    """Fetches and caches a service-account token via client_credentials grant."""

    def __init__(self, url: str, realm: str, client_id: str, client_secret: str):
        self._token_url = f"{url}/realms/{realm}/protocol/openid-connect/token"
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        # Refresh 30 s before actual expiry
        if self._access_token and time.time() < self._expires_at - 30:
            return self._access_token
        return self._refresh()

    def _refresh(self) -> str:
        resp = requests.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._expires_at = time.time() + data["expires_in"]
        print(f"[auth] token refreshed, expires in {data['expires_in']}s")
        return self._access_token