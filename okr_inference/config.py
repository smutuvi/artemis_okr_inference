"""Load task YAML and resolve runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_YAML = PACKAGE_ROOT / "config" / "tasks.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
DEFAULT_DOWNLOAD_DIR = PACKAGE_ROOT / "roboflow_datasets"

# Match-map colors (OpenCV BGR). Yellow "both" for visibility on dark canvas.
COLOR_ONLY_GT = (60, 200, 80)      # green
COLOR_ONLY_PRED = (60, 60, 230)    # red
COLOR_BOTH = (0, 220, 255)         # yellow
COLOR_BOTH_LEGEND = "#FFDC00"
COLOR_ONLY_GT_LEGEND = "#3cc850"
COLOR_ONLY_PRED_LEGEND = "#e03c3c"

# Left photo overlay (OpenCV draws on RGB arrays here → use RGB tuples).
COLOR_PHOTO_GT = (255, 105, 180)       # pink — ground truth
COLOR_PHOTO_PRED = (0, 210, 255)       # cyan — model predictions
COLOR_PHOTO_GT_LEGEND = "#FF69B4"
COLOR_PHOTO_PRED_LEGEND = "#00D2FF"
PHOTO_BOX_THICKNESS_GT = 3
PHOTO_BOX_THICKNESS_PRED = 4

COCO_STAT_NAMES = [
    "AP_50_95",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets1",
    "AR_maxDets10",
    "AR_maxDets100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


@dataclass
class ModelSpec:
    key: str
    label: str
    provider: str
    kind: str  # autolabel | roboflow
    model_type: str | None = None
    model_id: str | None = None


@dataclass
class TaskSpec:
    name: str
    gt_project: str
    gt_version: int
    classes: list[str]
    models: list[ModelSpec] = field(default_factory=list)


@dataclass
class RunConfig:
    workspace: str
    tasks: list[TaskSpec]
    match_iou: float = 0.50
    custom_conf: float = 0.40
    random_seed: int = 42
    download_format: str = "coco"
    force_redownload: bool = False
    sample_size: int | None = 20  # None => all images
    visualize: bool = False
    visualize_n: int = 1
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    download_dir: Path = field(default_factory=lambda: DEFAULT_DOWNLOAD_DIR)
    api_key: str = ""


def load_tasks_yaml(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_TASKS_YAML
    with open(path) as f:
        return yaml.safe_load(f)


def available_task_names(path: Path | None = None) -> list[str]:
    data = load_tasks_yaml(path)
    return list((data.get("tasks") or {}).keys())


def parse_task_selection(raw: str | None, all_names: list[str]) -> list[str]:
    """Accept 'all', comma-separated names, or repeated CLI values joined."""
    if not raw or raw.strip().lower() == "all":
        return list(all_names)
    names = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in all_names:
            raise ValueError(
                f"Unknown task '{name}'. Available: {', '.join(all_names)}"
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("No tasks selected.")
    return names


def parse_model_selection(raw: str | None) -> list[str] | None:
    """Return selected model keys, or None for all models on each task.

    Accepts: 'all' / empty → all models;
    'astra' | 'custom' | 'astra,custom' (keys from tasks.yaml);
    also aliases: 'autolabel' → astra, 'roboflow' → custom.
    """
    if not raw or raw.strip().lower() == "all":
        return None
    aliases = {
        "astra": "astra",
        "custom": "custom",
        "autolabel": "astra",
        "roboflow": "custom",
    }
    keys: list[str] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token not in aliases:
            raise ValueError(
                f"Unknown model '{part.strip()}'. "
                "Use: astra, custom, autolabel, roboflow, or all"
            )
        key = aliases[token]
        if key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError("No models selected.")
    return keys


def filter_task_models(tasks: list[TaskSpec], model_keys: list[str] | None) -> list[TaskSpec]:
    """Keep only models whose key is in model_keys (None = keep all)."""
    if model_keys is None:
        return tasks
    out: list[TaskSpec] = []
    for t in tasks:
        models = [m for m in t.models if m.key in model_keys]
        if not models:
            raise ValueError(
                f"Task '{t.name}' has none of the selected models {model_keys}. "
                f"Configured keys: {[m.key for m in t.models]}"
            )
        out.append(
            TaskSpec(
                name=t.name,
                gt_project=t.gt_project,
                gt_version=t.gt_version,
                classes=list(t.classes),
                models=models,
            )
        )
    return out


def build_task_specs(data: dict[str, Any], selected: list[str]) -> list[TaskSpec]:
    tasks_cfg = data.get("tasks") or {}
    out: list[TaskSpec] = []
    for name in selected:
        t = tasks_cfg[name]
        models = [
            ModelSpec(
                key=m["key"],
                label=m["label"],
                provider=m.get("provider", ""),
                kind=m["kind"],
                model_type=m.get("model_type"),
                model_id=m.get("model_id"),
            )
            for m in t.get("models", [])
        ]
        out.append(
            TaskSpec(
                name=name,
                gt_project=t["gt_project"],
                gt_version=int(t["gt_version"]),
                classes=list(t.get("classes") or []),
                models=models,
            )
        )
    return out


def resolve_api_key(cli_value: str | None = None) -> str:
    key = (cli_value or os.environ.get("ROBOFLOW_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "No Roboflow API key. Pass --api-key or set ROBOFLOW_API_KEY."
        )
    return key


def build_run_config(
    *,
    tasks: str | None = None,
    models: str | None = None,
    sample: int | None = 20,
    all_images: bool = False,
    visualize: bool = False,
    visualize_n: int = 1,
    tasks_yaml: Path | None = None,
    output_dir: Path | None = None,
    download_dir: Path | None = None,
    api_key: str | None = None,
    seed: int | None = None,
    match_iou: float | None = None,
    custom_conf: float | None = None,
    force_redownload: bool | None = None,
) -> RunConfig:
    data = load_tasks_yaml(tasks_yaml)
    defaults = data.get("defaults") or {}
    all_names = list((data.get("tasks") or {}).keys())
    selected = parse_task_selection(tasks, all_names)
    model_keys = parse_model_selection(models)
    task_specs = filter_task_models(build_task_specs(data, selected), model_keys)

    sample_size: int | None
    if all_images:
        sample_size = None
    else:
        sample_size = sample if sample is not None else 20

    return RunConfig(
        workspace=data.get("workspace", "cgiar-workspace"),
        tasks=task_specs,
        match_iou=float(match_iou if match_iou is not None else defaults.get("match_iou", 0.50)),
        custom_conf=float(
            custom_conf if custom_conf is not None else defaults.get("custom_conf", 0.40)
        ),
        random_seed=int(seed if seed is not None else defaults.get("random_seed", 42)),
        download_format=str(defaults.get("download_format", "coco")),
        force_redownload=bool(
            force_redownload
            if force_redownload is not None
            else defaults.get("force_redownload", False)
        ),
        sample_size=sample_size,
        visualize=bool(visualize),
        visualize_n=max(0, int(visualize_n)),
        output_dir=Path(output_dir or DEFAULT_OUTPUT_DIR),
        download_dir=Path(download_dir or DEFAULT_DOWNLOAD_DIR),
        api_key=resolve_api_key(api_key),
    )
