"""ROI helpers for detection and tracking pipelines."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO


def compute_roi(
    w: int, h: int, roi_half_width: Optional[int],
) -> Optional[Tuple[int, int, int, int]]:
    """Centered horizontal strip ROI; full frame when roi_half_width is falsy."""
    if not roi_half_width:
        return None
    cx = w // 2
    return (max(0, cx - roi_half_width), 0, min(w - 1, cx + roi_half_width), h - 1)


def compute_center_roi_by_ratio(
    frame_shape: Tuple[int, int],
    roi_width_ratio: float,
    roi_height_ratio: float,
) -> Tuple[int, int, int, int]:
    """Return centered ROI by width/height ratio as (x1, y1, x2, y2)."""
    h, w = frame_shape
    roi_w = int(w * roi_width_ratio)
    roi_h = int(h * roi_height_ratio)
    x1 = max((w - roi_w) // 2, 0)
    y1 = max((h - roi_h) // 2, 0)
    x2 = min(x1 + roi_w, w)
    y2 = min(y1 + roi_h, h)
    return (x1, y1, x2, y2)


def crop_frame_with_roi(
    frame: np.ndarray, roi: Tuple[int, int, int, int]
) -> np.ndarray:
    """Crop frame using ROI tuple (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def yolo_detections_with_roi(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    nms: float,
    roi: Optional[Tuple[int, int, int, int]],
) -> sv.Detections:
    """Run ROI crop detection and restore global xyxy coordinates."""
    if roi is not None:
        x0, y0, x1, y1 = roi
        crop = frame[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size == 0:
            return sv.Detections.empty()
        res = model(crop, conf=conf, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res).with_nms(threshold=nms)
        if len(dets) > 0:
            off = np.array([x0, y0, x0, y0], dtype=np.float32)
            dets.xyxy = dets.xyxy + off
        return dets
    res = model(frame, conf=conf, verbose=False)[0]
    return sv.Detections.from_ultralytics(res).with_nms(threshold=nms)
