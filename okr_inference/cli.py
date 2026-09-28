"""CLI for Artemis OKR Astra-vs-custom evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from okr_inference.config import (
    DEFAULT_TASKS_YAML,
    available_task_names,
    build_run_config,
)
from okr_inference.evaluate import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    names = available_task_names()
    p = argparse.ArgumentParser(
        prog="okr_inference",
        description=(
            "Evaluate GPT-6 Astra Autolabel vs custom Roboflow models on "
            "Artemis OKR GT datasets. Select tasks, sample or all images, "
            "and optionally save playground-style visualizations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tasks",
        default="flowers,pods",
        help=(
            f"Comma-separated task names, or 'all'. Available: {', '.join(names)}. "
            "Example: flowers  |  flowers,pods  |  all"
        ),
    )
    p.add_argument(
        "--models",
        default="all",
        help=(
            "Which model(s) to run per task: astra, custom, or both (all). "
            "Aliases: autolabel→astra, roboflow→custom. "
            "Example: --models astra  |  --models custom  |  --models astra,custom"
        ),
    )
    sample = p.add_mutually_exclusive_group()
    sample.add_argument(
        "--sample",
        type=int,
        default=20,
        metavar="N",
        help="Number of seeded images per task (ignored if --all).",
    )
    sample.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every image in each selected task's GT split.",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Save playground-style panels (detections | match map | mAP card).",
    )
    p.add_argument(
        "--visualize-n",
        type=int,
        default=1,
        metavar="N",
        help="How many sample images per model to visualize (requires --visualize).",
    )
    p.add_argument("--api-key", default=None, help="Roboflow API key (else $ROBOFLOW_API_KEY).")
    p.add_argument(
        "--tasks-yaml",
        type=Path,
        default=DEFAULT_TASKS_YAML,
        help="Path to tasks YAML.",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Where to write CSVs/viz.")
    p.add_argument("--download-dir", type=Path, default=None, help="Where to cache GT downloads.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    p.add_argument("--match-iou", type=float, default=None, help="IoU for match-map / micro P-R.")
    p.add_argument("--custom-conf", type=float, default=None, help="Confidence floor for custom models.")
    p.add_argument(
        "--force-redownload",
        action="store_true",
        help="Delete matching local GT folders before downloading.",
    )
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available task names from the YAML and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        for name in available_task_names(args.tasks_yaml):
            print(name)
        return 0

    cfg = build_run_config(
        tasks=args.tasks,
        models=args.models,
        sample=args.sample,
        all_images=args.all,
        visualize=args.visualize,
        visualize_n=args.visualize_n,
        tasks_yaml=args.tasks_yaml,
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        api_key=args.api_key,
        seed=args.seed,
        match_iou=args.match_iou,
        custom_conf=args.custom_conf,
        force_redownload=args.force_redownload,
    )
    run_evaluation(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
