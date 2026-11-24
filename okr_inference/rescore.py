"""Re-score an existing run's checkpoints at one or more confidence floors.

Does not re-run inference. Uses checkpoints/*.jsonl + the same GT download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO

from okr_inference.coco_io import download_gt, find_coco_split
from okr_inference.config import (
    DEFAULT_TASKS_YAML,
    PACKAGE_ROOT,
    TaskSpec,
    build_run_config,
    resolve_api_key,
)
from okr_inference.evaluate import load_checkpoint, _ckpt_path
from okr_inference.metrics import (
    apply_class_aliases,
    coco_eval_bbox_and_segm,
    map_pred_classes,
    norm_name,
    per_image_match_stats,
    resolve_cats,
)


DEFAULT_OKR_CSV = PACKAGE_ROOT / "config" / "OKR_4.3.csv"


def _task_from_cfg(cfg, task_name: str) -> TaskSpec:
    for t in cfg.tasks:
        if t.name == task_name:
            return t
    raise ValueError(
        f"Task {task_name!r} not in config. Available: {[t.name for t in cfg.tasks]}"
    )


def rescore_task(
    *,
    run_dir: Path,
    task: TaskSpec,
    confs: list[float],
    workspace: str,
    api_key: str,
    download_dir: Path,
    split: str,
    download_format: str = "coco",
) -> pd.DataFrame:
    ds_location = download_gt(
        workspace=workspace,
        api_key=api_key,
        task=task,
        download_dir=download_dir,
        download_format=download_format,
        force_redownload=False,
    )
    ann, _img_dir, used_split = find_coco_split(
        ds_location, preferred=split, require_preferred=False
    )
    print(f"GT split: {used_split}  (requested {split})")
    coco = COCO(ann)
    name_to_id = resolve_cats(coco, task.classes)
    if not name_to_id:
        raise ValueError(f"No GT classes for {task.name}: {task.classes}")

    id_to_name = {i: n for n, i in name_to_id.items()}
    ckpt = load_checkpoint(_ckpt_path(run_dir, task.name))
    if not ckpt:
        raise FileNotFoundError(f"No checkpoint for {task.name} under {run_dir}")

    rows = []
    for conf in confs:
        model_dets: dict[str, list] = {}
        both_sum = pred_sum = gt_sum = 0
        raw_pred_classes: set[str] = set()

        for img_id, rec in ckpt.items():
            info = coco.loadImgs(img_id)[0] if img_id in set(coco.getImgIds()) else None
            # GT for scored classes
            if info is None:
                continue
            anns = coco.loadAnns(
                coco.getAnnIds(imgIds=img_id, catIds=list(name_to_id.values()))
            )
            gt_pairs = [
                (a["bbox"], coco.loadCats([a["category_id"]])[0]["name"]) for a in anns
            ]
            gt_sum += len(gt_pairs)

            for model_key, dets in (rec.get("dets") or {}).items():
                kept = []
                for d in dets:
                    score = float(d.get("score", 0.0))
                    if score < conf:
                        continue
                    # checkpoint may store class name or only category_id
                    cls = d.get("class")
                    if cls is None:
                        cls = id_to_name.get(int(d["category_id"]))
                    if cls is None:
                        continue
                    raw_pred_classes.add(str(cls))
                    mapped = apply_class_aliases(
                        [{"class": cls, "bbox": d["bbox"], "score": score}],
                        task.class_aliases,
                    )
                    mapped = map_pred_classes(mapped, name_to_id)
                    if not mapped:
                        continue
                    p = mapped[0]
                    kept.append(
                        {
                            "bbox": d["bbox"],
                            "score": score,
                            "class": p["class"],
                            "image_id": img_id,
                            "category_id": name_to_id[p["class"]],
                            **{
                                k: d[k]
                                for k in ("points", "segmentation", "mask")
                                if d.get(k) is not None
                            },
                        }
                    )

                stats = per_image_match_stats(gt_pairs, kept, 0.5)
                both_sum += stats["n_both"]
                pred_sum += stats["n_pred"]
                model_dets.setdefault(model_key, []).extend(
                    [
                        {
                            "image_id": p["image_id"],
                            "category_id": p["category_id"],
                            "bbox": p["bbox"],
                            "score": p["score"],
                            **{
                                k: p[k]
                                for k in ("points", "segmentation", "mask")
                                if p.get(k) is not None
                            },
                        }
                        for p in kept
                    ]
                )

        # one model key expected (custom)
        model_key = next(iter(model_dets), "custom")
        dets = model_dets.get(model_key, [])
        sample_ids = sorted(ckpt.keys())
        coco_stats = coco_eval_bbox_and_segm(
            coco,
            sample_ids,
            name_to_id,
            dets,
            run_segm=task.run_segm,
        )
        rows.append(
            {
                "task": task.name,
                "conf": conf,
                "expected_objects": gt_sum,
                "predicted_objects": pred_sum,
                "matched_both": both_sum,
                "micro_precision_iou50": (both_sum / pred_sum) if pred_sum else None,
                "micro_recall_iou50": (both_sum / gt_sum) if gt_sum else None,
                "pred_classes_seen": ",".join(sorted(raw_pred_classes)),
                "gt_classes": ",".join(sorted(name_to_id.keys())),
                **coco_stats,
            }
        )
        print(
            f"  conf={conf:.3f}  bbox_AP@50={coco_stats.get('AP_50')}  "
            f"pred={pred_sum} match={both_sum}  "
            f"pred_classes={sorted(raw_pred_classes)}"
        )

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-score checkpoints at multiple confidence floors (no re-infer)."
    )
    p.add_argument("--run-dir", type=Path, required=True, help="Existing outputs/... run folder")
    p.add_argument(
        "--task",
        required=True,
        help="Task name (e.g. cowpea_plant_stand). Must match checkpoint file stem.",
    )
    p.add_argument(
        "--conf-sweep",
        default="0.01,0.10,0.25,0.40,0.50",
        help="Comma-separated confidence floors to evaluate.",
    )
    p.add_argument("--okr-csv", nargs="?", const=str(DEFAULT_OKR_CSV), default=None)
    p.add_argument("--tasks-yaml", type=Path, default=DEFAULT_TASKS_YAML)
    p.add_argument("--split", default="valid", choices=["test", "valid", "train"])
    p.add_argument("--download-dir", type=Path, default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to write the sweep table (default: RUN/task_conf_sweep.csv).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    confs = [float(x) for x in args.conf_sweep.split(",") if x.strip()]
    cfg = build_run_config(
        tasks=args.task,
        models="custom",
        all_images=True,
        tasks_yaml=args.tasks_yaml,
        download_dir=args.download_dir,
        api_key=resolve_api_key(args.api_key),
        split=args.split,
        okr_csv=Path(args.okr_csv) if args.okr_csv else None,
        okr_ready_only=False,
    )
    task = _task_from_cfg(cfg, args.task)
    df = rescore_task(
        run_dir=args.run_dir,
        task=task,
        confs=confs,
        workspace=cfg.workspace,
        api_key=cfg.api_key,
        download_dir=cfg.download_dir,
        split=cfg.split,
    )
    out = args.out_csv or (args.run_dir / f"{args.task}_conf_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\n📄 Wrote {out}")
    cols = [
        c
        for c in (
            "conf",
            "AP_50",
            "AP_50_95",
            "AP_50_segm",
            "expected_objects",
            "predicted_objects",
            "matched_both",
            "micro_precision_iou50",
            "micro_recall_iou50",
            "pred_classes_seen",
            "gt_classes",
        )
        if c in df.columns
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
