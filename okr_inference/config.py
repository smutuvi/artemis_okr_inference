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
    resume_dir: Path | None = None  # reuse this run folder and skip done images
    run_name: str | None = None  # custom folder name under output_dir


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


def parse_model_selection(
    raw: str | None,
    available_keys: list[str] | None = None,
) -> tuple[list[str] | None, str | None]:
    """Parse --model / --models.

    Returns (keys, adhoc_model_type):
      - keys is None → run every model listed on each task
      - keys is a list → filter task models to those keys
      - adhoc_model_type set → inject a one-off Autolabel modelType
        (used when the user passes a raw type like gpt-6-sol-boxes)

    Accepts known keys (astra, custom, sol, …), aliases, comma lists, or a
    raw Autolabel modelType string.
    """
    if not raw or raw.strip().lower() == "all":
        return None, None

    aliases = {
        "astra": "astra",
        "custom": "custom",
        "sol": "sol",
        "gpt-sol": "sol",
        "gpt6-sol": "sol",
        "gpt-6-sol": "sol",
        "autolabel": "astra",
        "roboflow": "custom",
    }
    available = set(available_keys or [])
    keys: list[str] = []
    adhoc: str | None = None

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # Single raw Autolabel modelType (e.g. gpt-6-sol-boxes)
    if len(parts) == 1:
        token = parts[0]
        low = token.lower()
        if low in aliases:
            return [aliases[low]], None
        if low in available:
            return [low], None
        if "gpt-" in low or low.endswith("-boxes") or "/" in low:
            return None, token  # preserve original casing for API

    for part in parts:
        low = part.lower()
        if low in aliases:
            key = aliases[low]
        elif low in available:
            key = low
        else:
            known = sorted(set(aliases) | available | {"all"})
            raise ValueError(
                f"Unknown model '{part}'. Use one of: {', '.join(known)} "
                "or a raw Autolabel modelType like gpt-6-sol-boxes"
            )
        if key not in keys:
            keys.append(key)

    if not keys and not adhoc:
        raise ValueError("No models selected.")
    return keys, adhoc


def inject_autolabel_model(tasks: list[TaskSpec], model_type: str) -> list[TaskSpec]:
    """Replace each task's models with a single Autolabel model of model_type."""
    low = model_type.lower()
    if "sol" in low:
        key, label = "sol", "GPT-6 Sol"
    elif "astra" in low:
        key, label = "astra", "GPT-6 Astra"
    else:
        key, label = "adhoc", model_type

    model = ModelSpec(
        key=key,
        label=label,
        provider="Roboflow Autolabel",
        kind="autolabel",
        model_type=model_type,
    )
    return [
        TaskSpec(
            name=t.name,
            gt_project=t.gt_project,
            gt_version=t.gt_version,
            classes=list(t.classes),
            models=[model],
        )
        for t in tasks
    ]


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


def load_dotenv_file(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (no extra dependency).

    Looks in the project root by default. Existing environment variables win.
    """
    path = path or (PACKAGE_ROOT / ".env")
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        os.environ.setdefault(k, v)


def resolve_api_key(cli_value: str | None = None) -> str:
    load_dotenv_file()
    key = (cli_value or os.environ.get("ROBOFLOW_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "No Roboflow API key. Pass --api-key, set ROBOFLOW_API_KEY, "
            "or add ROBOFLOW_API_KEY=... to .env in the project root."
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
    resume_dir: Path | None = None,
    run_name: str | None = None,
) -> RunConfig:
    data = load_tasks_yaml(tasks_yaml)
    defaults = data.get("defaults") or {}
    all_names = list((data.get("tasks") or {}).keys())
    selected = parse_task_selection(tasks, all_names)
    task_specs = build_task_specs(data, selected)
    available_model_keys = sorted(
        {m.key for t in task_specs for m in t.models}
    )
    model_keys, adhoc_type = parse_model_selection(models, available_model_keys)
    if adhoc_type:
        task_specs = inject_autolabel_model(task_specs, adhoc_type)
    else:
        task_specs = filter_task_models(task_specs, model_keys)

    sample_size: int | None
    if all_images:
        sample_size = None
    else:
        sample_size = sample if sample is not None else 20

    resume = Path(resume_dir) if resume_dir else None
    if resume is not None and not resume.is_dir():
        raise ValueError(f"--resume path is not a directory: {resume}")

    name = (run_name or "").strip() or None
    if name is not None:
        # Keep it a single path segment (no nested paths / traversal)
        name = Path(name).name
        if not name or name in (".", ".."):
            raise ValueError(f"Invalid --run-name: {run_name!r}")

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
        resume_dir=resume,
        run_name=name,
    )
