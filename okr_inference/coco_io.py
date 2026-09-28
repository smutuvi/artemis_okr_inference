"""Download GT datasets and locate COCO splits."""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

from roboflow import Roboflow

from okr_inference.config import TaskSpec


def find_coco_split(dataset_location: str | Path) -> tuple[str, str, str]:
    dataset_location = str(dataset_location)
    for split in ("valid", "test", "train"):
        p = os.path.join(dataset_location, split, "_annotations.coco.json")
        if os.path.exists(p):
            return p, os.path.join(dataset_location, split), split

    candidates: list[str] = []
    for pattern in (
        "**/*annotations*.json",
        "**/_annotations.coco.json",
        "**/annotations.json",
    ):
        candidates.extend(glob.glob(os.path.join(dataset_location, pattern), recursive=True))
    candidates = sorted(set(candidates), key=lambda p: (("coco" not in p.lower()), len(p)))
    for ann_path in candidates:
        try:
            with open(ann_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                img_dir = os.path.dirname(ann_path)
                return ann_path, img_dir, os.path.basename(img_dir)
        except Exception:
            continue
    raise FileNotFoundError(f"No COCO annotations under {dataset_location}")


def download_gt(
    *,
    workspace: str,
    api_key: str,
    task: TaskSpec,
    download_dir: Path,
    download_format: str = "coco",
    force_redownload: bool = False,
):
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {task.gt_project}/{task.gt_version} as {download_format}…")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(task.gt_project)
    version_obj = project.version(task.gt_version)

    if force_redownload:
        short = task.gt_project.lower().replace("_", "-")
        for path in glob.glob(str(download_dir / "*")):
            base = os.path.basename(path).lower()
            if not os.path.isdir(path):
                continue
            if (short in base or task.gt_project.lower() in base) and (
                str(task.gt_version) in base or base.endswith(f"-{task.gt_version}")
            ):
                print(f"  Removing stale folder: {path}")
                shutil.rmtree(path, ignore_errors=True)

    # Roboflow SDK downloads relative to CWD; pin location via chdir.
    cwd = os.getcwd()
    try:
        os.chdir(download_dir)
        ds = version_obj.download(download_format, overwrite=True)
    finally:
        os.chdir(cwd)

    location = Path(ds.location)
    if not location.is_absolute():
        location = download_dir / location
    print(f"  location={location}")
    return location
