import requests
from typing import Optional, Dict
BASE_URL = "http://127.0.0.1:8000"

def api_request(method: str, path: str, params: dict = None, json: dict = None):
    url = f"{BASE_URL}{path}"
    try:
        if method.upper() == "GET":
            r = requests.get(url, params=params, timeout=10)
        else:
            r = requests.post(url, json=json, timeout=10)
        r.raise_for_status()
        return r
    except Exception as e:
        # return None on errors; pages will show message
        return None
