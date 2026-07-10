"""
api_client.py - Thin authenticated client for the CITIZEN API.

Logs in with the face-gate device account (role Security) via
api/v1/auth/login.php, holds the JWT, and re-authenticates
automatically when a request comes back 401 (token expired).
"""

import configparser
import os

import requests


class CitizenAPI:
    def __init__(self, config_path="config.ini"):
        cfg = configparser.ConfigParser()
        if not cfg.read(config_path):
            raise FileNotFoundError(f"Cannot read {config_path}")

        self.base = cfg["api"]["base_url"].rstrip("/")          # .../citizen/api/v1
        self.root = self.base.rsplit("/api/", 1)[0]              # .../citizen
        self.email = cfg["api"]["email"]
        self.password = cfg["api"]["password"]
        self.timeout = int(cfg["api"].get("timeout", "8"))
        self.token = None
        self.cfg = cfg

    # ── auth ────────────────────────────────────────────────────────────
    def login(self):
        r = requests.post(
            f"{self.base}/auth/login.php",
            json={"email": self.email, "password": self.password},
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        # Adjust the line below if your login.php nests the token differently
        self.token = (
            body.get("data", {}).get("token")
            or body.get("token")
            or body.get("data", {}).get("jwt")
        )
        if not self.token:
            raise RuntimeError(f"login.php returned no token: {body}")
        print(f"[api] Logged in as {self.email}")

    def _headers(self):
        if not self.token:
            self.login()
        return {"Authorization": f"Bearer {self.token}"}

    def _retry_on_401(self, do_request):
        resp = do_request()
        if resp.status_code == 401:
            print("[api] Token rejected/expired - re-authenticating...")
            self.login()
            resp = do_request()
        return resp

    # ── verbs ───────────────────────────────────────────────────────────
    def get(self, path, params=None):
        return self._retry_on_401(
            lambda: requests.get(f"{self.base}/{path}", params=params,
                                 headers=self._headers(), timeout=self.timeout)
        )

    def post_json(self, path, payload):
        return self._retry_on_401(
            lambda: requests.post(f"{self.base}/{path}", json=payload,
                                  headers=self._headers(), timeout=self.timeout)
        )

    def post_multipart(self, path, data, files):
        return self._retry_on_401(
            lambda: requests.post(f"{self.base}/{path}", data=data, files=files,
                                  headers=self._headers(), timeout=self.timeout)
        )

    def patch_json(self, path, payload):
        return self._retry_on_401(
            lambda: requests.patch(f"{self.base}/{path}", json=payload,
                                   headers=self._headers(), timeout=self.timeout)
        )

    def download(self, web_path, dest):
        """Download a project-relative file, e.g. uploads/profiles/x.jpg"""
        url = f"{self.root}/{web_path.lstrip('/')}"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as f:
            f.write(r.content)
        return dest
