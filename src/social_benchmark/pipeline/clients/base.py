from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ApiClientError(RuntimeError):
    pass


@dataclass(slots=True)
class HttpJsonClient:
    base_url: str
    user_agent: str = "social-benchmark/0.1"
    timeout_seconds: int = 20
    retries: int = 2
    min_delay_seconds: float = 0.0

    def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = self._build_url(path, params)
        merged_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if headers:
            merged_headers.update(headers)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.min_delay_seconds:
                time.sleep(self.min_delay_seconds)
            try:
                request = Request(url, headers=merged_headers)
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    if raw == "null":
                        return None
                    return json.loads(raw)
            except HTTPError as exc:
                last_error = exc
                if exc.code in {403, 404, 422}:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(0.5 * (2**attempt))
        raise ApiClientError(f"GET {url} failed: {last_error}") from last_error

    def _build_url(self, path: str, params: dict[str, Any] | None) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            base = path
        else:
            base = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        clean_params = {key: value for key, value in (params or {}).items() if value is not None}
        if not clean_params:
            return base
        return f"{base}?{urlencode(clean_params, doseq=True)}"

