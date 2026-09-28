"""Playground-style visualization panels."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from okr_inference.config import (
    COLOR_BOTH,
    COLOR_BOTH_LEGEND,
    COLOR_ONLY_GT,
    COLOR_ONLY_GT_LEGEND,
    COLOR_ONLY_PRED,
    COLOR_ONLY_PRED_LEGEND,
    COLOR_PHOTO_GT,
    COLOR_PHOTO_GT_LEGEND,
    COLOR_PHOTO_PRED,
    COLOR_PHOTO_PRED_LEGEND,
    PHOTO_BOX_THICKNESS_GT,
    PHOTO_BOX_THICKNESS_PRED,
)
from okr_inference.metrics import match_boxes


def _iter_bboxes(boxes):
    """Yield (x, y, w, h) from pred dicts or (bbox, class) GT pairs."""
    for item in boxes:
        if isinstance(item, dict):
            x, y, w, h = map(int, item["bbox"])
        else:
            (x, y, w, h), _cls = item
            x, y, w, h = map(int, (x, y, w, h))
        yield x, y, w, h


def draw_photo_overlay(
    img_rgb,
    gt_pairs=None,
    preds=None,
    *,
    gt_color=COLOR_PHOTO_GT,
    pred_color=COLOR_PHOTO_PRED,
    gt_thickness=PHOTO_BOX_THICKNESS_GT,
    pred_thickness=PHOTO_BOX_THICKNESS_PRED,
):
    """Draw GT (pink) then predictions (thick cyan) on the photo."""
    out = img_rgb.copy()
    for x, y, w, h in _iter_bboxes(gt_pairs or []):
        cv2.rectangle(out, (x, y), (x + w, y + h), gt_color, gt_thickness)
    for x, y, w, h in _iter_bboxes(preds or []):
        cv2.rectangle(out, (x, y), (x + w, y + h), pred_color, pred_thickness)

    # Corner legend swatches
    y0 = 8
    if gt_pairs:
        cv2.rectangle(out, (8, y0), (26, y0 + 18), gt_color, -1)
        cv2.putText(
            out, "GT", (32, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )
        y0 += 26
    if preds is not None:
        cv2.rectangle(out, (8, y0), (26, y0 + 18), pred_color, -1)
        cv2.putText(
            out, "pred", (32, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return out


def draw_match_map(img_shape, both, only_gt, only_pred):
    h, w = img_shape[:2]
    canvas = np.full((h, w, 3), 18, dtype=np.uint8)

    def rect(bbox, color, thickness=3):
        x, y, bw, bh = map(int, bbox)
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, thickness)

    for item in both:
        gb = item[0][0]
        rect(gb, COLOR_BOTH)
    for gb, _ in only_gt:
        rect(gb, COLOR_ONLY_GT)
    for p in only_pred:
        rect(p["bbox"], COLOR_ONLY_PRED)
    return canvas


def playground_panel(
    img_rgb,
    model_meta: dict,
    gt_pairs,
    preds,
    map50,
    expected_n: int,
    predicted_n: int,
    match_iou: float = 0.5,
    save_path: Path | None = None,
    show: bool = False,
):
    both, only_gt, only_pred = match_boxes(gt_pairs, preds, match_iou)
    det_img = draw_photo_overlay(img_rgb, gt_pairs=gt_pairs, preds=preds)
    match_img = draw_match_map(img_rgb.shape, both, only_gt, only_pred)

    fig = plt.figure(figsize=(16, 5.2), facecolor="#111111")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.85], wspace=0.08)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(det_img)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for s in ax0.spines.values():
        s.set_color("#333")
    ax0.legend(
        handles=[
            mpatches.Patch(color=COLOR_PHOTO_GT_LEGEND, label="ground truth"),
            mpatches.Patch(color=COLOR_PHOTO_PRED_LEGEND, label="prediction"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
    )

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    ax1.set_xticks([])
    ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_color("#333")
    ax1.legend(
        handles=[
            mpatches.Patch(color=COLOR_ONLY_GT_LEGEND, label="only in expected"),
            mpatches.Patch(color=COLOR_ONLY_PRED_LEGEND, label="only in model"),
            mpatches.Patch(color=COLOR_BOTH_LEGEND, label="both"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
    )

    ax2 = fig.add_subplot(gs[2])
    ax2.set_facecolor("#111111")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)

    map_txt = f"{map50 * 100:.1f}%" if map50 is not None else "n/a"
    color = "#3DDC97" if (map50 is not None and map50 >= 0.7) else "#E85D5D"
    ax2.text(
        0.02, 0.92, model_meta["label"], color="white", fontsize=15,
        fontweight="bold", transform=ax2.transAxes,
    )
    ax2.text(
        0.02, 0.82, model_meta["provider"], color="#9aa0a6", fontsize=10,
        transform=ax2.transAxes,
    )
    ax2.text(
        0.98, 0.92, "OBJECT DETECTION", color="#9aa0a6", fontsize=9,
        ha="right", transform=ax2.transAxes,
    )
    ax2.text(0.02, 0.55, "mAP@50", color="#9aa0a6", fontsize=11, transform=ax2.transAxes)
    ax2.text(
        0.02, 0.28, map_txt, color=color, fontsize=36, fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.02, 0.12, f"{expected_n} expected objects", color="#c5c8ce",
        fontsize=11, transform=ax2.transAxes,
    )
    ax2.text(
        0.02, 0.04, f"{predicted_n} predicted objects", color="#c5c8ce",
        fontsize=11, transform=ax2.transAxes,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  🖼  saved viz → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
