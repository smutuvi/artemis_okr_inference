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


def _empty_coco_stats() -> dict[str, float | None]:
    return {n: None for n in COCO_STAT_NAMES}


def segmentation_from_det(d: dict) -> list | dict | None:
    """Return a COCO segmentation (polygon list or RLE dict) from a det, if any."""
    seg = d.get("segmentation")
    if isinstance(seg, dict) and seg.get("counts") is not None:
        return seg
    if isinstance(seg, list) and seg:
        # Already COCO polygons [[x,y,...], ...] or a flat ring
        if isinstance(seg[0], (int, float)):
            return [seg] if len(seg) >= 6 else None
        return seg
    points = d.get("points")
    if isinstance(points, list) and len(points) >= 3:
        flat: list[float] = []
        for pt in points:
            if isinstance(pt, dict) and "x" in pt and "y" in pt:
                flat.extend([float(pt["x"]), float(pt["y"])])
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                flat.extend([float(pt[0]), float(pt[1])])
            else:
                return None
        return [flat] if len(flat) >= 6 else None
    return None


def gt_has_segmentation(
    coco_gt: COCO,
    sample_ids: list[int],
    cat_ids: list[int],
) -> bool:
    """True if any GT ann in the sample has a usable segmentation."""
    ann_ids = coco_gt.getAnnIds(imgIds=sample_ids, catIds=cat_ids)
    for a in coco_gt.loadAnns(ann_ids):
        seg = a.get("segmentation")
        if not seg:
            continue
        if isinstance(seg, dict) and seg.get("counts") is not None:
            return True
        if isinstance(seg, list) and len(seg) > 0:
            return True
    return False


def coco_eval_full(
    coco_gt: COCO,
    sample_ids: list[int],
    name_to_id: dict[str, int],
    dets: list[dict],
    *,
    iou_type: str = "bbox",
) -> dict[str, float | None]:
    empty = _empty_coco_stats()
    if not dets:
        return empty

    if iou_type == "bbox":
        coco_dets = [
            {
                "image_id": d["image_id"],
                "category_id": d["category_id"],
                "bbox": d["bbox"],
                "score": d["score"],
            }
            for d in dets
        ]
    elif iou_type == "segm":
        coco_dets = []
        for d in dets:
            seg = segmentation_from_det(d)
            if seg is None:
                continue
            item = {
                "image_id": d["image_id"],
                "category_id": d["category_id"],
                "segmentation": seg,
                "score": d["score"],
            }
            if d.get("bbox") is not None:
                item["bbox"] = d["bbox"]
            coco_dets.append(item)
        if not coco_dets:
            return empty
    else:
        raise ValueError(f"Unsupported iou_type: {iou_type}")

    coco_dt = coco_gt.loadRes(copy.deepcopy(coco_dets))
    ev = COCOeval(coco_gt, coco_dt, iou_type)
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


def coco_eval_bbox_and_segm(
    coco_gt: COCO,
    sample_ids: list[int],
    name_to_id: dict[str, int],
    dets: list[dict],
    *,
    run_segm: bool = False,
) -> dict[str, float | None]:
    """Bbox COCO stats, plus mask stats as ``*_segm`` when ``run_segm`` is True."""
    out = coco_eval_full(coco_gt, sample_ids, name_to_id, dets, iou_type="bbox")
    if not run_segm:
        return out

    cat_ids = list(name_to_id.values())
    if not gt_has_segmentation(coco_gt, sample_ids, cat_ids):
        print("  ⚠️ GT has no segmentations — skipping mask mAP (segm)")
        for n in COCO_STAT_NAMES:
            out[f"{n}_segm"] = None
        return out

    n_mask = sum(1 for d in dets if segmentation_from_det(d) is not None)
    print(f"  Mask mAP (segm): {n_mask}/{len(dets)} preds have polygons/masks")
    segm = coco_eval_full(coco_gt, sample_ids, name_to_id, dets, iou_type="segm")
    for n, v in segm.items():
        out[f"{n}_segm"] = v
    return out
