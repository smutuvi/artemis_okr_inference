"""Astra Autolabel + Roboflow Serverless inference."""

from __future__ import annotations

import base64
import time
from typing import Any, Callable

import requests

# Transient network / server failures that should be retried on long runs.
_RETRYABLE = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.SSLError,
)


def file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def parse_boxes(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    preds = (
        payload.get("predictions")
        or payload.get("annotations")
        or payload.get("labels")
        or []
    )
    if not preds and isinstance(payload.get("result"), dict):
        preds = payload["result"].get("predictions", [])
    if not preds and isinstance(payload.get("data"), dict):
        preds = payload["data"].get("predictions", [])

    out = []
    for p in preds:
        if not isinstance(p, dict):
            continue
        if all(k in p for k in ("x", "y", "width", "height")):
            w, h = float(p["width"]), float(p["height"])
            x = float(p["x"]) - w / 2.0
            y = float(p["y"]) - h / 2.0
        elif all(k in p for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = map(float, (p["x1"], p["y1"], p["x2"], p["y2"]))
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
        elif "bbox" in p and len(p["bbox"]) == 4:
            x, y, w, h = map(float, p["bbox"])
        else:
            continue
        out.append(
            {
                "bbox": [x, y, w, h],
                "score": float(p.get("confidence", p.get("score", 1.0))),
                "class": p.get("class") or p.get("label") or "object",
            }
        )
    return out


def _with_retries(
    label: str,
    fn: Callable[[], Any],
    *,
    retries: int = 5,
    base_delay: float = 2.0,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except _RETRYABLE as e:
            last_err = e
            if attempt >= retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"      ⚠️ {label} attempt {attempt}/{retries} failed "
                f"({type(e).__name__}: {e}); retry in {delay:.0f}s"
            )
            time.sleep(delay)
        except requests.exceptions.HTTPError as e:
            # Retry 429 / 5xx
            status = getattr(getattr(e, "response", None), "status_code", None)
            last_err = e
            if status not in (429, 500, 502, 503, 504) or attempt >= retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"      ⚠️ {label} HTTP {status} attempt {attempt}/{retries}; "
                f"retry in {delay:.0f}s"
            )
            time.sleep(delay)
    print(f"      ❌ {label} failed after {retries} attempts: {last_err}")
    return None


def infer_astra(
    *,
    workspace: str,
    api_key: str,
    gt_project_id: str,
    image_path: str,
    ontology: dict[str, str],
    model_type: str = "gpt-6-astra-boxes",
    timeout: int = 420,
    retries: int = 5,
) -> list[dict]:
    url = (
        f"https://api.roboflow.com/{workspace}/{gt_project_id}/autolabel/preview"
        f"?api_key={api_key}"
    )
    body = {
        "modelType": model_type,
        "image": {"type": "base64", "value": file_b64(image_path)},
        "ontology": ontology,
    }

    def _once():
        r = requests.post(url, json=body, timeout=timeout)
        print(f"      Autolabel[{model_type}] HTTP {r.status_code}")
        if r.status_code in (429, 500, 502, 503, 504):
            r.raise_for_status()
        if not r.ok:
            print("      ", r.text[:300])
            return []
        try:
            return parse_boxes(r.json())
        except Exception:
            print("      non-JSON:", r.text[:300])
            return []

    result = _with_retries(f"Autolabel[{model_type}]", _once, retries=retries)
    return result if isinstance(result, list) else []


def infer_custom(
    *,
    api_key: str,
    model_id: str,
    image_path: str,
    confidence: float = 0.40,
    timeout: int = 180,
    retries: int = 5,
) -> list[dict]:
    # Two Roboflow model ID styles:
    #   legacy "project/<version>"           -> multipart upload, api_key in query
    #   new    "workspace/model-name"        -> base64 body, Bearer auth
    versioned = model_id.rsplit("/", 1)[-1].isdigit()
    url = f"https://serverless.roboflow.com/{model_id}"

    def _post():
        if versioned:
            with open(image_path, "rb") as f:
                return requests.post(
                    url,
                    params={"api_key": api_key, "confidence": confidence},
                    files={"file": f},
                    timeout=timeout,
                )
        return requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=file_b64(image_path),
            timeout=timeout,
        )

    def _once():
        r = _post()
        print(f"      Custom[{model_id}] HTTP {r.status_code}")
        if r.status_code in (429, 500, 502, 503, 504):
            r.raise_for_status()
        if not r.ok:
            print("      ", r.text[:300])
            return []
        try:
            data = r.json()
        except Exception:
            print("      non-JSON:", r.text[:300])
            return []
        return [p for p in parse_boxes(data) if p["score"] >= confidence]

    result = _with_retries(f"Custom[{model_id}]", _once, retries=retries)
    return result if isinstance(result, list) else []
