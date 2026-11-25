"""One safe way to POST JSON to an HTTP API, shared by the LLM call and the embeddings call.

Standard library only. Redirects are never followed (urllib would forward the Authorization header to
wherever they point), and every failure becomes a readable message that names the .env entry to fix.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class ServiceError(RuntimeError):
    """A problem talking to an outside service, phrased so it can be shown to the user as is."""


@dataclass(frozen=True)
class Service:
    """How a service is described to the user in error messages."""
    label: str       # e.g. "LLM service"
    key_var: str     # e.g. "LLM_API_KEY"
    url_var: str     # e.g. "LLM_BASE_URL"
    model_var: str   # e.g. "LLM_MODEL"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def error_text(payload) -> str:
    """The human-readable message inside an error body (formats differ between services)."""
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            if payload.get(key):
                return error_text(payload[key])
        return ""
    if isinstance(payload, list):
        return "; ".join(filter(None, (error_text(item) for item in payload)))
    return str(payload or "")


def _http_error(exc: urllib.error.HTTPError, service: Service, error: type[ServiceError]) -> ServiceError:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except OSError:
        raw = ""
    finally:
        exc.close()
    try:
        detail = error_text(json.loads(raw))
    except ValueError:
        detail = ""
    detail = (detail or raw).strip()[:300]
    suffix = f" ({detail})" if detail else ""
    if exc.code in (401, 403):
        return error(f"The {service.label} rejected the API key (HTTP {exc.code}). "
                     f"Check {service.key_var} in your .env file.{suffix}")
    if exc.code == 404:
        return error(f"The {service.label} answered 'not found' (HTTP 404). Check {service.model_var} (is the "
                     f"model name right?) and {service.url_var} in your .env file.{suffix}")
    if 300 <= exc.code < 400:
        return error(f"The {service.label} redirected the request (HTTP {exc.code}). Check {service.url_var} "
                     "in your .env file (usually it needs https:// or a /v1 ending).")
    if exc.code == 429:
        return error(f"{service.label} rate limit or quota reached (HTTP 429). Wait a moment and try again.{suffix}")
    return error(f"{service.label} error (HTTP {exc.code}){suffix}")


def post_json(url: str, headers: dict, body: dict, service: Service, timeout: float,
              error: type[ServiceError] = ServiceError) -> dict:
    """POST `body` as JSON and return the parsed JSON reply. Raises `error` (a ServiceError) on any failure."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:   # must come before URLError, which it subclasses
        raise _http_error(exc, service, error) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise error(f"The {service.label} did not answer within {timeout:g} seconds.") from exc
        raise error(f"Could not reach the {service.label} at {url} ({exc.reason}). Check {service.url_var} and "
                    "your connection (or that your local server is running).") from exc
    except TimeoutError as exc:
        raise error(f"The {service.label} did not answer within {timeout:g} seconds.") from exc
    except ValueError as exc:   # JSONDecodeError / UnicodeDecodeError: the reply was not JSON
        raise error(f"The {service.label} did not reply with JSON. Check that {service.url_var} points at an "
                    "OpenAI-compatible API (usually ending in /v1).") from exc
