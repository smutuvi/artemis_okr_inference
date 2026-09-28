"""Matching helpers and COCO evaluation."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from okr_inference.config import COCO_STAT_NAMES


def norm_name(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("_", "-")


def resolve_cats(coco: COCO, wanted: list[str]) -> dict[str, int]:
    cats = coco.loadCats(coco.getCatIds())
    print("  Categories:", [(c["id"], c["name"]) for c in cats])
    out: dict[str, int] = {}
    for w in wanted:
        hit = next((c for c in cats if norm_name(c["name"]) == norm_name(w)), None)
        if hit:
            out[hit["name"]] = hit["id"]
        else:
            print(f"  ⚠️ GT missing class: {w}")
    if not out:
        non_bg = [c for c in cats if not str(c["name"]).startswith("__")]
        if non_bg:
            print(f"  Falling back to: {[c['name'] for c in non_bg]}")
            out = {c["name"]: c["id"] for c in non_bg}
    return out


def map_pred_classes(preds: list[dict], name_to_id: dict[str, int]) -> list[dict]:
    by_norm = {norm_name(n): n for n in name_to_id}
    out = []
    for p in preds:
        exact = by_norm.get(norm_name(p["class"]))
        if exact is None:
            for n in name_to_id:
                if norm_name(n) in norm_name(p["class"]) or norm_name(p["class"]) in norm_name(n):
                    exact = n
                    break
        if exact is None:
            continue
        q = dict(p)
        q["class"] = exact
        out.append(q)
    return out


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def match_boxes(gt_boxes, pred_boxes, iou_thr: float = 0.5):
    """gt_boxes: list[(bbox, class)]; pred_boxes: list[dict]."""
    gt, pr = list(gt_boxes), list(pred_boxes)
    both, used_g, used_p = [], set(), set()
    pairs = []
    for gi, (gb, gc) in enumerate(gt):
        for pi, p in enumerate(pr):
            if norm_name(gc) != norm_name(p["class"]):
                continue
            pairs.append((iou_xywh(gb, p["bbox"]), gi, pi))
    pairs.sort(reverse=True)
    for iou, gi, pi in pairs:
        if iou < iou_thr:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        both.append((gt[gi], pr[pi], iou))
    only_gt = [gt[i] for i in range(len(gt)) if i not in used_g]
    only_pred = [pr[i] for i in range(len(pr)) if i not in used_p]
    return both, only_gt, only_pred


def per_image_match_stats(gt_pairs, preds, iou_thr: float = 0.5) -> dict[str, Any]:
    both, only_gt, only_pred = match_boxes(gt_pairs, preds, iou_thr)
    n_gt, n_pred, n_both = len(gt_pairs), len(preds), len(both)
    return {
        "n_pred": n_pred,
        "n_both": n_both,
        "n_only_gt": len(only_gt),
        "n_only_pred": len(only_pred),
        "precision_iou50": (n_both / n_pred) if n_pred else None,
        "recall_iou50": (n_both / n_gt) if n_gt else None,
        "mean_matched_iou": float(np.mean([b[2] for b in both])) if both else None,
    }


def coco_eval_full(
    coco_gt: COCO,
    sample_ids: list[int],
    name_to_id: dict[str, int],
    dets: list[dict],
) -> dict[str, float | None]:
    empty = {n: None for n in COCO_STAT_NAMES}
    if not dets:
        return empty
    coco_dt = coco_gt.loadRes(copy.deepcopy(dets))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = sample_ids
    ev.params.catIds = list(name_to_id.values())
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    out: dict[str, float | None] = {}
    for i, name in enumerate(COCO_STAT_NAMES):
        val = float(ev.stats[i])
        out[name] = None if np.isnan(val) else val
    return out
