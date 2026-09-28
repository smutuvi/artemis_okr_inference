"""Astra Autolabel + Roboflow Serverless inference."""

from __future__ import annotations

import base64
from typing import Any

import requests


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


def infer_astra(
    *,
    workspace: str,
    api_key: str,
    gt_project_id: str,
    image_path: str,
    ontology: dict[str, str],
    model_type: str = "gpt-6-astra-boxes",
    timeout: int = 420,
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
    r = requests.post(url, json=body, timeout=timeout)
    print(f"      Astra HTTP {r.status_code}")
    if not r.ok:
        print("      ", r.text[:300])
        return []
    try:
        return parse_boxes(r.json())
    except Exception:
        print("      non-JSON:", r.text[:300])
        return []


def infer_custom(
    *,
    api_key: str,
    model_id: str,
    image_path: str,
    confidence: float = 0.40,
    timeout: int = 180,
) -> list[dict]:
    url = (
        f"https://serverless.roboflow.com/{model_id}"
        f"?api_key={api_key}&confidence={confidence}"
    )
    with open(image_path, "rb") as f:
        r = requests.post(url, files={"file": f}, timeout=timeout)
    print(f"      Custom[{model_id}] HTTP {r.status_code}")
    if not r.ok:
        print("      ", r.text[:300])
        return []
    try:
        data = r.json()
    except Exception:
        print("      non-JSON:", r.text[:300])
        return []
    return [p for p in parse_boxes(data) if p["score"] >= confidence]
