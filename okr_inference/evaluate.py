"""Run selected OKR tasks: Astra vs custom, CSV + optional viz."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from okr_inference.coco_io import download_gt, find_coco_split
from okr_inference.config import ModelSpec, RunConfig, TaskSpec
from okr_inference.infer import infer_astra, infer_custom
from okr_inference.metrics import (
    coco_eval_full,
    map_pred_classes,
    norm_name,
    per_image_match_stats,
    resolve_cats,
)
from okr_inference.viz import playground_panel


def seeded_ids(img_ids, k: int | None, name: str, seed: int) -> list[int]:
    ids = list(img_ids)
    if k is None or k >= len(ids):
        print(f"  sample=ALL n={len(ids)}")
        return sorted(ids)
    s = seed + sum(ord(c) for c in name)
    rng = random.Random(s)
    picked = sorted(rng.sample(ids, min(k, len(ids))))
    print(f"  sample seed={s} n={len(picked)} ids={picked}")
    return picked


def _image_path(img_dir: str, ds_location: Path, file_name: str) -> str:
    path = os.path.join(img_dir, file_name)
    if os.path.exists(path):
        return path
    alt = os.path.join(ds_location, file_name)
    return alt if os.path.exists(alt) else path


def _run_model_infer(
    cfg: RunConfig,
    task: TaskSpec,
    model: ModelSpec,
    image_path: str,
    ontology: dict[str, str],
) -> list[dict]:
    if model.kind == "autolabel":
        return infer_astra(
            workspace=cfg.workspace,
            api_key=cfg.api_key,
            gt_project_id=task.gt_project,
            image_path=image_path,
            ontology=ontology,
            model_type=model.model_type or "gpt-6-astra-boxes",
        )
    if model.kind == "roboflow":
        if not model.model_id:
            raise ValueError(f"Model {model.key} missing model_id")
        return infer_custom(
            api_key=cfg.api_key,
            model_id=model.model_id,
            image_path=image_path,
            confidence=cfg.custom_conf,
        )
    raise ValueError(f"Unknown model kind: {model.kind}")


def _ckpt_path(run_dir: Path, task_name: str) -> Path:
    return run_dir / "checkpoints" / f"{task_name}.jsonl"


def load_checkpoint(path: Path) -> dict[int, dict]:
    """Return image_id -> record for completed images (last write wins)."""
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[int(rec["image_id"])] = rec
    return done


def append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_task(cfg: RunConfig, task: TaskSpec, run_dir: Path) -> dict | None:
    print("\n" + "#" * 72)
    print(f"# TASK: {task.name}  GT={task.gt_project}/{task.gt_version}")
    print("#" * 72)

    try:
        ds_location = download_gt(
            workspace=cfg.workspace,
            api_key=cfg.api_key,
            task=task,
            download_dir=cfg.download_dir,
            download_format=cfg.download_format,
            force_redownload=cfg.force_redownload,
        )
    except Exception as e:
        print(f"❌ GT download failed: {e}")
        return None

    ann, img_dir, split = find_coco_split(ds_location)
    print("GT split:", split)
    coco = COCO(ann)
    name_to_id = resolve_cats(coco, task.classes)
    if not name_to_id:
        print("❌ no GT classes")
        return None

    ontology = {n: n for n in name_to_id}
    id_by_norm = {norm_name(n): i for n, i in name_to_id.items()}
    sample_ids = seeded_ids(
        coco.getImgIds(), cfg.sample_size, task.name, cfg.random_seed
    )

    ckpt = _ckpt_path(run_dir, task.name)
    done = load_checkpoint(ckpt)
    if done:
        print(f"  Resume: {len(done)}/{len(sample_ids)} images already in {ckpt}")

    model_dets: dict[str, list] = {m.key: [] for m in task.models}
    per_image_rows: list[dict] = []
    # Keep only enough viz frames for --visualize N (prefer earliest sample order)
    per_image_viz: list[dict] = []
    model_keys = [m.key for m in task.models]

    for i, img_id in enumerate(sample_ids):
        info = coco.loadImgs(img_id)[0]
        path = _image_path(img_dir, ds_location, info["file_name"])
        anns = coco.loadAnns(
            coco.getAnnIds(imgIds=img_id, catIds=list(name_to_id.values()))
        )
        gt_pairs = [
            (a["bbox"], coco.loadCats([a["category_id"]])[0]["name"]) for a in anns
        ]

        # ---- resume from checkpoint ----
        if img_id in done:
            rec = done[img_id]
            row = rec["row"]
            per_image_rows.append(row)
            for k in model_keys:
                for p in rec.get("dets", {}).get(k, []):
                    model_dets[k].append(p)
            if (
                cfg.visualize
                and len(per_image_viz) < cfg.visualize_n
                and os.path.exists(path)
            ):
                img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                preds_by_key = {}
                for k in model_keys:
                    preds_by_key[k] = [
                        {
                            "bbox": d["bbox"],
                            "score": d["score"],
                            "class": next(
                                (
                                    n
                                    for n, cid in name_to_id.items()
                                    if cid == d["category_id"]
                                ),
                                "object",
                            ),
                        }
                        for d in rec.get("dets", {}).get(k, [])
                    ]
                per_image_viz.append(
                    {
                        "gt": gt_pairs,
                        "img": img_rgb,
                        "preds": preds_by_key,
                        "image_name": info["file_name"],
                    }
                )
            if (i + 1) % 50 == 0 or i == 0:
                print(f"\n[{i+1}/{len(sample_ids)}] {info['file_name']}  (cached)")
            continue

        print(f"\n[{i+1}/{len(sample_ids)}] {info['file_name']}  GT={len(gt_pairs)}")

        row = {
            "task": task.name,
            "image_id": img_id,
            "image_name": info["file_name"],
            "width": info.get("width"),
            "height": info.get("height"),
            "gt_count": len(gt_pairs),
            "gt_classes": ",".join(sorted({c for _, c in gt_pairs})),
        }

        if not os.path.exists(path):
            print("  ⚠️ missing image")
            for m in task.models:
                k = m.key
                for col in (
                    "n_pred", "n_both", "n_only_gt", "n_only_pred",
                    "precision_iou50", "recall_iou50", "mean_matched_iou",
                    "pred_classes", "mean_score",
                ):
                    row[f"{k}_{col}"] = None
            per_image_rows.append(row)
            append_checkpoint(
                ckpt, {"image_id": img_id, "row": row, "dets": {k: [] for k in model_keys}}
            )
            continue

        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        viz = {"gt": gt_pairs, "img": img_rgb, "preds": {}, "image_name": info["file_name"]}
        dets_this: dict[str, list] = {k: [] for k in model_keys}

        for m in task.models:
            k = m.key
            try:
                preds = _run_model_infer(cfg, task, m, path, ontology)
            except Exception as e:
                print(f"    ❌ {k} infer error (skipping image preds): {e}")
                preds = []
            preds = map_pred_classes(preds, name_to_id)
            print(f"    {k}: {len(preds)} preds")
            viz["preds"][k] = preds

            stats = per_image_match_stats(gt_pairs, preds, cfg.match_iou)
            row[f"{k}_n_pred"] = stats["n_pred"]
            row[f"{k}_n_both"] = stats["n_both"]
            row[f"{k}_n_only_gt"] = stats["n_only_gt"]
            row[f"{k}_n_only_pred"] = stats["n_only_pred"]
            row[f"{k}_precision_iou50"] = stats["precision_iou50"]
            row[f"{k}_recall_iou50"] = stats["recall_iou50"]
            row[f"{k}_mean_matched_iou"] = stats["mean_matched_iou"]
            row[f"{k}_pred_classes"] = (
                ",".join(sorted({p["class"] for p in preds})) if preds else ""
            )
            row[f"{k}_mean_score"] = (
                float(np.mean([p["score"] for p in preds])) if preds else None
            )

            for p in preds:
                det = {
                    "image_id": img_id,
                    "category_id": id_by_norm[norm_name(p["class"])],
                    "bbox": p["bbox"],
                    "score": p["score"],
                }
                model_dets[k].append(det)
                dets_this[k].append(det)

        per_image_rows.append(row)
        if cfg.visualize and len(per_image_viz) < cfg.visualize_n:
            per_image_viz.append(viz)

        append_checkpoint(
            ckpt, {"image_id": img_id, "row": row, "dets": dets_this}
        )
        # Lightweight progress CSV so a kill mid-run still leaves something readable
        pd.DataFrame(per_image_rows).to_csv(
            run_dir / f"{task.name}_per_image.csv", index=False
        )

    df_img = pd.DataFrame(per_image_rows)
    img_csv = run_dir / f"{task.name}_per_image.csv"
    df_img.to_csv(img_csv, index=False)
    print(f"\n📄 Wrote {img_csv}")

    expected_total = int(df_img["gt_count"].fillna(0).sum()) if len(df_img) else 0
    model_rows = []
    results = {
        "task": task.name,
        "sample_ids": sample_ids,
        "models": {},
        "df_img": df_img,
    }

    viz_dir = run_dir / "visualizations" / task.name
    for m in task.models:
        key = m.key
        coco_stats = coco_eval_full(coco, sample_ids, name_to_id, model_dets[key])
        pred_total = len(model_dets[key])
        map50 = coco_stats.get("AP_50")

        both_sum = int(df_img[f"{key}_n_both"].fillna(0).sum()) if len(df_img) else 0
        only_gt_sum = int(df_img[f"{key}_n_only_gt"].fillna(0).sum()) if len(df_img) else 0
        only_pred_sum = (
            int(df_img[f"{key}_n_only_pred"].fillna(0).sum()) if len(df_img) else 0
        )

        mrow = {
            "task": task.name,
            "model_key": key,
            "model_label": m.label,
            "provider": m.provider,
            "model_id": m.model_id or m.model_type,
            "gt_project": task.gt_project,
            "gt_version": task.gt_version,
            "n_images": len(sample_ids),
            "sample_size": cfg.sample_size if cfg.sample_size is not None else "all",
            "random_seed": cfg.random_seed,
            "match_iou": cfg.match_iou,
            "custom_conf": cfg.custom_conf if m.kind == "roboflow" else None,
            "expected_objects": expected_total,
            "predicted_objects": pred_total,
            "matched_both": both_sum,
            "only_in_gt": only_gt_sum,
            "only_in_pred": only_pred_sum,
            "micro_precision_iou50": both_sum / pred_total if pred_total else None,
            "micro_recall_iou50": both_sum / expected_total if expected_total else None,
            **coco_stats,
        }
        model_rows.append(mrow)
        results["models"][key] = mrow

        print(
            f"\n[{task.name}] {m.label}: mAP@50={map50}  "
            f"expected={expected_total} predicted={pred_total}"
        )

        if cfg.visualize and cfg.visualize_n > 0 and per_image_viz:
            n = min(cfg.visualize_n, len(per_image_viz))
            for vi in range(n):
                r = per_image_viz[vi]
                safe = Path(r["image_name"]).stem
                out = viz_dir / f"{key}__{safe}.png"
                playground_panel(
                    r["img"],
                    {"label": m.label, "provider": m.provider},
                    r["gt"],
                    r["preds"].get(key, []),
                    map50,
                    expected_n=len(r["gt"]),
                    predicted_n=len(r["preds"].get(key, [])),
                    match_iou=cfg.match_iou,
                    save_path=out,
                    show=False,
                )

    df_models = pd.DataFrame(model_rows)
    models_csv = run_dir / f"{task.name}_model_metrics.csv"
    df_models.to_csv(models_csv, index=False)
    print(f"📄 Wrote {models_csv}")
    results["df_models"] = df_models
    return results


def write_summary(all_results: list[dict], run_dir: Path) -> pd.DataFrame | None:
    frames = [r["df_models"] for r in all_results if r is not None]
    if not frames:
        return None

    df_summary = pd.concat(frames, ignore_index=True)
    front = [
        "task", "model_key", "model_label", "provider", "model_id",
        "n_images", "expected_objects", "predicted_objects",
        "matched_both", "only_in_gt", "only_in_pred",
        "micro_precision_iou50", "micro_recall_iou50",
        "AP_50_95", "AP_50", "AP_75",
        "AP_small", "AP_medium", "AP_large",
        "AR_maxDets1", "AR_maxDets10", "AR_maxDets100",
        "AR_small", "AR_medium", "AR_large",
        "gt_project", "gt_version", "sample_size", "random_seed",
        "match_iou", "custom_conf",
    ]
    cols = [c for c in front if c in df_summary.columns] + [
        c for c in df_summary.columns if c not in front
    ]
    df_summary = df_summary[cols]

    summary_csv = run_dir / "summary_model_performance.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n📄 Wrote {summary_csv}")

    board = df_summary[
        [
            "task", "model_label", "provider",
            "AP_50", "AP_50_95", "AP_75", "AR_maxDets100",
            "expected_objects", "predicted_objects",
            "matched_both", "micro_precision_iou50", "micro_recall_iou50",
        ]
    ].copy()
    board_csv = run_dir / "summary_leaderboard.csv"
    board.to_csv(board_csv, index=False)
    print(f"📄 Wrote {board_csv}")

    print("\n" + "#" * 72)
    print("# SUMMARY — model performance")
    print("#" * 72)
    display_cols = [
        "task", "model_label", "AP_50", "AP_50_95", "AP_75",
        "expected_objects", "predicted_objects", "matched_both",
        "micro_precision_iou50", "micro_recall_iou50",
    ]
    print(df_summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n# Best AP@50 per task")
    for task_name, g in df_summary.groupby("task"):
        if g["AP_50"].notna().any():
            best = g.loc[g["AP_50"].idxmax()]
            print(
                f"  {task_name}: {best['model_label']}  AP@50={best['AP_50']:.3f}  "
                f"AP@50:95={best['AP_50_95']:.3f}"
            )
    return df_summary


def run_evaluation(cfg: RunConfig) -> Path:
    if cfg.resume_dir is not None:
        run_dir = Path(cfg.resume_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print("Resuming into:", run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_tag = "-".join(t.name for t in cfg.tasks) or "none"
        run_dir = cfg.output_dir / f"{stamp}_{task_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print("Output dir:", run_dir)

    cfg.download_dir.mkdir(parents=True, exist_ok=True)

    print("Tasks:", [t.name for t in cfg.tasks])
    print(
        "Models:",
        {t.name: [m.key for m in t.models] for t in cfg.tasks},
    )
    print(
        "Sample:",
        "ALL" if cfg.sample_size is None else cfg.sample_size,
        "| visualize:",
        cfg.visualize,
        f"(n={cfg.visualize_n})" if cfg.visualize else "",
    )

    results = []
    for task in cfg.tasks:
        results.append(run_task(cfg, task, run_dir))
    results = [r for r in results if r]
    write_summary(results, run_dir)
    print("\nAll outputs in:", run_dir)
    return run_dir
