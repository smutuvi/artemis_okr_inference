"""Load OKR trait rows from CSV (model_path + data_path Roboflow URLs)."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from urllib.parse import urlparse, unquote

from okr_inference.config import ModelSpec, TaskSpec

# Traits that should keep instance masks and run COCO segm mAP.
_SEGM_TRAIT_HINTS = (
    "flower",
    "pod",
    "panicle",
    "pearl",
    "grain",
    "inst",
    "seg",
)


def slugify_task(crop: str, trait: str) -> str:
    raw = f"{crop}_{trait}".strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    return raw.strip("_") or "task"


def parse_classes(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def _path_parts(url: str) -> list[str]:
    u = urlparse((url or "").strip())
    path = unquote(u.path or "").strip("/")
    return [p for p in path.split("/") if p]


def parse_model_path(url: str) -> str:
    """Convert a Roboflow model page URL to a serverless model_id.

    Supports:
      .../workspace/project/models/project/VERSION
      .../workspace/project/models/cgiar-workspace/model-name
      .../workspace/project/VERSION   (versioned project = trained model)
    """
    if not url or not str(url).strip():
        raise ValueError("Empty model_path")
    parts = _path_parts(url)
    if len(parts) < 2:
        raise ValueError(f"Cannot parse model_path: {url}")

    if "models" in parts:
        i = parts.index("models")
        rest = parts[i + 1 :]
        if not rest:
            raise ValueError(f"model_path has /models/ but no id: {url}")
        # workspace-scoped deployed model: cgiar-workspace/foo-rfdetr-...
        if len(rest) >= 2 and rest[0].endswith("-workspace"):
            return "/".join(rest[:2])
        if len(rest) >= 2 and re.fullmatch(r"\d+", rest[-1]):
            return f"{rest[-2]}/{rest[-1]}"
        if len(rest) == 1:
            return rest[0]
        return "/".join(rest)

    # .../workspace/project/version
    if len(parts) >= 3 and re.fullmatch(r"\d+", parts[-1]):
        return f"{parts[-2]}/{parts[-1]}"

    raise ValueError(f"Cannot parse model_path: {url}")


def parse_data_path(url: str) -> tuple[str, int]:
    """Convert a Roboflow dataset version URL to (project_id, version)."""
    if not url or not str(url).strip():
        raise ValueError("Empty data_path")
    parts = _path_parts(url)
    # expect: workspace / project / version
    if len(parts) < 3 or not re.fullmatch(r"\d+", parts[-1]):
        raise ValueError(
            f"data_path must look like .../workspace/project/VERSION, got: {url}"
        )
    return parts[-2], int(parts[-1])


def trait_wants_segm(trait: str) -> bool:
    t = (trait or "").lower()
    if "plant stand" in t or t.replace("_", " ") == "plant stand":
        return False
    return any(h in t for h in _SEGM_TRAIT_HINTS)


def load_okr_csv(
    path: Path | str,
    *,
    ready_only: bool = True,
    require_paths: bool = True,
) -> list[TaskSpec]:
    """Build TaskSpecs from an OKR_4.3-style CSV.

    Requires columns: crop, trait, model_path, data_path, classes_covered.
    Optional: dataset_status, active_model.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    tasks: list[TaskSpec] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        for i, row in enumerate(reader, start=2):
            crop = (row.get("crop") or "").strip()
            trait = (row.get("trait") or "").strip()
            if not crop and not trait:
                continue

            status = (row.get("dataset_status") or "").strip().lower()
            if ready_only and status and status != "ready":
                print(f"  skip row {i} {crop}/{trait}: dataset_status={status!r}")
                continue

            model_url = (row.get("model_path") or "").strip()
            data_url = (row.get("data_path") or "").strip()
            classes = parse_classes(row.get("classes_covered"))

            if require_paths and (not model_url or not data_url):
                print(
                    f"  skip row {i} {crop}/{trait}: missing model_path and/or data_path"
                )
                continue
            if not classes:
                print(f"  skip row {i} {crop}/{trait}: empty classes_covered")
                continue

            try:
                model_id = parse_model_path(model_url)
                gt_project, gt_version = parse_data_path(data_url)
            except ValueError as e:
                print(f"  skip row {i} {crop}/{trait}: {e}")
                continue

            name = slugify_task(crop, trait)
            label = (row.get("active_model") or "").strip() or f"Custom {name}"
            label = f"{label} ({model_id})"
            run_segm = trait_wants_segm(trait)

            tasks.append(
                TaskSpec(
                    name=name,
                    gt_project=gt_project,
                    gt_version=gt_version,
                    classes=classes,
                    models=[
                        ModelSpec(
                            key="custom",
                            label=label,
                            provider="Roboflow Serverless",
                            kind="roboflow",
                            model_id=model_id,
                        )
                    ],
                    run_segm=run_segm,
                )
            )
            print(
                f"  OKR task {name}: GT={gt_project}/{gt_version}  "
                f"model={model_id}  classes={classes}  segm={run_segm}"
            )

    if not tasks:
        raise ValueError(f"No runnable OKR tasks found in {path}")
    return tasks
