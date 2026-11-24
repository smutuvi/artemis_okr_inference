"""CLI for Artemis OKR Astra-vs-custom evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from okr_inference.config import (
    DEFAULT_TASKS_YAML,
    PACKAGE_ROOT,
    available_task_names,
    build_run_config,
)
from okr_inference.evaluate import run_evaluation

DEFAULT_OKR_CSV = PACKAGE_ROOT / "config" / "OKR_4.3.csv"


def build_parser() -> argparse.ArgumentParser:
    names = available_task_names()
    p = argparse.ArgumentParser(
        prog="okr_inference",
        description=(
            "Evaluate custom Roboflow models on Artemis OKR GT datasets "
            "(tasks.yaml or OKR CSV with model_path + data_path). "
            "Select tasks, sample or all images, optional visualizations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tasks",
        default="all",
        help=(
            f"Comma-separated task names, or 'all'. YAML tasks: {', '.join(names)}. "
            "With --okr-csv, names are crop_trait slugs (e.g. cowpea_pods)."
        ),
    )
    p.add_argument(
        "--okr-csv",
        nargs="?",
        const=str(DEFAULT_OKR_CSV),
        default=None,
        metavar="PATH",
        help=(
            "Load tasks from an OKR CSV (model_path + data_path). "
            f"Omit PATH to use {DEFAULT_OKR_CSV.name}."
        ),
    )
    p.add_argument(
        "--okr-include-non-ready",
        action="store_true",
        help="With --okr-csv, also include rows whose dataset_status is not 'ready'.",
    )
    p.add_argument(
        "--models",
        "--model",
        dest="models",
        default="custom",
        help=(
            "Which model(s) to run: custom (default), astra, sol, comma list, or all. "
            "Aliases: gpt-6-sol→sol, autolabel→astra, roboflow→custom. "
            "Or pass a raw Autolabel modelType, e.g. gpt-6-sol-boxes. "
            "Example: --model custom  |  --models astra,custom"
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
        nargs="?",
        const=1,
        default=0,
        type=int,
        metavar="N",
        help=(
            "Save playground-style panels for N images per model "
            "(detections | match map | mAP card). "
            "Omit N to use 1: --visualize   or   --visualize 5"
        ),
    )
    p.add_argument(
        "--visualize-n",
        type=int,
        default=None,
        metavar="N",
        help="Deprecated alias for --visualize N (same meaning).",
    )
    p.add_argument("--api-key", default=None, help="Roboflow API key (else $ROBOFLOW_API_KEY).")
    p.add_argument(
        "--tasks-yaml",
        type=Path,
        default=DEFAULT_TASKS_YAML,
        help="Path to tasks YAML.",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Where to write CSVs/viz.")
    p.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help=(
            "Custom folder name under --output-dir (e.g. plant_stand_custom_all). "
            "Default is <timestamp>_<tasks>."
        ),
    )
    p.add_argument("--download-dir", type=Path, default=None, help="Where to cache GT downloads.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    p.add_argument("--match-iou", type=float, default=None, help="IoU for match-map / micro P-R.")
    p.add_argument("--custom-conf", type=float, default=None, help="Confidence floor for custom models.")
    p.add_argument(
        "--max-detections",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max detections per image for custom models (Roboflow default 300). "
            "Use 2000–3000 for dense sorghum panicle/pearl."
        ),
    )
    p.add_argument(
        "--force-redownload",
        action="store_true",
        help="Delete matching local GT folders before downloading.",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "valid", "train"],
        help="GT COCO split to evaluate on (default: test).",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Continue an interrupted run in this output folder "
            "(skips images already saved under checkpoints/*.jsonl)."
        ),
    )
    p.add_argument(
        "--task-classes",
        action="append",
        default=None,
        metavar="TASK=CLASS,...",
        help=(
            "Override GT classes for a task. Repeatable. "
            "Example: --task-classes pods=Fruit_pod,Plant_Bean"
        ),
    )
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available task names from the YAML and exit.",
    )
    p.add_argument(
        "--pack-jsonl",
        action="store_true",
        help=(
            "After the run, zip checkpoints + *_predictions_with_masks.jsonl "
            "to ~/ <run>_jsonl.zip for Jupyter download (GitHub-safe push still "
            "excludes these files)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        if args.okr_csv:
            from okr_inference.okr_csv import load_okr_csv

            for t in load_okr_csv(
                args.okr_csv, ready_only=not args.okr_include_non_ready
            ):
                print(
                    f"{t.name}\tGT={t.gt_project}/{t.gt_version}\t"
                    f"model={t.models[0].model_id}\tclasses={t.classes}"
                )
        else:
            for name in available_task_names(args.tasks_yaml):
                print(name)
        return 0

    # --visualize [N]  (default N=1); --visualize-n kept as alias
    viz_n = args.visualize_n if args.visualize_n is not None else args.visualize
    if viz_n < 0:
        parser.error("--visualize N must be >= 0")

    cfg = build_run_config(
        tasks=args.tasks,
        models=args.models,
        sample=args.sample,
        all_images=args.all,
        visualize=viz_n > 0,
        visualize_n=viz_n,
        tasks_yaml=args.tasks_yaml,
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        api_key=args.api_key,
        seed=args.seed,
        match_iou=args.match_iou,
        custom_conf=args.custom_conf,
        max_detections=args.max_detections,
        force_redownload=args.force_redownload,
        resume_dir=args.resume,
        run_name=args.run_name,
        split=args.split,
        task_classes=args.task_classes,
        okr_csv=Path(args.okr_csv) if args.okr_csv else None,
        okr_ready_only=not args.okr_include_non_ready,
    )
    run_dir = run_evaluation(cfg)
    if args.pack_jsonl:
        from okr_inference.pack_outputs import pack_run_jsonl

        try:
            pack_run_jsonl(run_dir)
        except FileNotFoundError as e:
            print(f"⚠️ --pack-jsonl: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
