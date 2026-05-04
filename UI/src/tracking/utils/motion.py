"""Bounding-box warping for Roboflow trackers camera motion compensation (CMC).

Uses ``MotionEstimator.update()`` -> ``CoordinatesTransformation`` and applies
``rel_to_abs`` / ``abs_to_rel`` to axis-aligned box corners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from trackers.motion.transformation import CoordinatesTransformation


def warp_xyxy_rel_to_abs(xyxy: np.ndarray, coord: "CoordinatesTransformation") -> np.ndarray:
    """Map boxes from current frame to stabilized (first-frame) coordinates."""
    xyxy = np.asarray(xyxy, dtype=np.float64)
    n = len(xyxy)
    if n == 0:
        return xyxy.astype(np.float32)
    corners = np.empty((n * 4, 2), dtype=np.float64)
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i]
        j = i * 4
        corners[j] = (x1, y1)
        corners[j + 1] = (x2, y1)
        corners[j + 2] = (x2, y2)
        corners[j + 3] = (x1, y2)
    warped = coord.rel_to_abs(corners)
    out = np.empty((n, 4), dtype=np.float32)
    for i in range(n):
        block = warped[i * 4 : (i + 1) * 4]
        out[i, 0] = float(block[:, 0].min())
        out[i, 1] = float(block[:, 1].min())
        out[i, 2] = float(block[:, 0].max())
        out[i, 3] = float(block[:, 1].max())
    return out


def warp_xyxy_abs_to_rel(xyxy: np.ndarray, coord: "CoordinatesTransformation") -> np.ndarray:
    """Map boxes from stabilized coordinates back to the current frame."""
    xyxy = np.asarray(xyxy, dtype=np.float64)
    n = len(xyxy)
    if n == 0:
        return xyxy.astype(np.float32)
    corners = np.empty((n * 4, 2), dtype=np.float64)
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i]
        j = i * 4
        corners[j] = (x1, y1)
        corners[j + 1] = (x2, y1)
        corners[j + 2] = (x2, y2)
        corners[j + 3] = (x1, y2)
    warped = coord.abs_to_rel(corners)
    out = np.empty((n, 4), dtype=np.float32)
    for i in range(n):
        block = warped[i * 4 : (i + 1) * 4]
        out[i, 0] = float(block[:, 0].min())
        out[i, 1] = float(block[:, 1].min())
        out[i, 2] = float(block[:, 0].max())
        out[i, 3] = float(block[:, 1].max())
    return out


def warp_detections_rel_to_abs(detections: sv.Detections, coord: "CoordinatesTransformation") -> sv.Detections:
    if len(detections) == 0:
        return detections
    new_xyxy = warp_xyxy_rel_to_abs(detections.xyxy, coord)
    return sv.Detections(
        xyxy=new_xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id,
    )


def warp_detections_abs_to_rel(detections: sv.Detections, coord: "CoordinatesTransformation") -> sv.Detections:
    if len(detections) == 0:
        return detections
    new_xyxy = warp_xyxy_abs_to_rel(detections.xyxy, coord)
    out = sv.Detections(
        xyxy=new_xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id,
    )
    if detections.tracker_id is not None:
        out.tracker_id = detections.tracker_id
    return out


def warp_xywh_list_rel_to_abs(
    items: List[Tuple[List[float], float, int]],
    coord: "CoordinatesTransformation",
) -> List[Tuple[List[float], float, int]]:
    """DeepSORT-style ``([x,y,w,h], conf, class)`` tuples in frame coords -> stabilized."""
    out: List[Tuple[List[float], float, int]] = []
    for xywh, conf, cid in items:
        x, y, w, h = xywh
        xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float64)
        stab = warp_xyxy_rel_to_abs(xyxy, coord)[0]
        x1, y1, x2, y2 = stab.tolist()
        out.append(([x1, y1, x2 - x1, y2 - y1], conf, cid))
    return out


def warp_ltrb_abs_to_rel(ltrb: np.ndarray, coord: "CoordinatesTransformation") -> np.ndarray:
    """Single ``ltrb`` (x1,y1,x2,y2) in stabilized coords -> current frame."""
    x1, y1, x2, y2 = np.asarray(ltrb, dtype=np.float64).reshape(-1).tolist()[:4]
    xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float64)
    return warp_xyxy_abs_to_rel(xyxy, coord)[0]


def is_reasonable_motion_transform(
    coord: "CoordinatesTransformation",
    frame_w: int,
    frame_h: int,
    max_center_shift_ratio: float = 0.40,
    max_scale_change_ratio: float = 0.40,
) -> bool:
    """Reject pathological homographies that produce unstable boxes."""
    corners = np.array(
        [
            [0.0, 0.0],
            [frame_w - 1.0, 0.0],
            [frame_w - 1.0, frame_h - 1.0],
            [0.0, frame_h - 1.0],
        ],
        dtype=np.float64,
    )
    warped = np.asarray(coord.rel_to_abs(corners), dtype=np.float64)
    if warped.shape != (4, 2) or not np.isfinite(warped).all():
        return False

    orig_center = corners.mean(axis=0)
    warped_center = warped.mean(axis=0)
    center_shift = float(np.linalg.norm(warped_center - orig_center))
    max_shift = float(np.hypot(frame_w, frame_h) * max_center_shift_ratio)
    if center_shift > max_shift:
        return False

    warped_w = float(warped[:, 0].max() - warped[:, 0].min())
    warped_h = float(warped[:, 1].max() - warped[:, 1].min())
    if warped_w <= 1.0 or warped_h <= 1.0:
        return False

    sx = warped_w / max(frame_w, 1)
    sy = warped_h / max(frame_h, 1)
    min_s = 1.0 - max_scale_change_ratio
    max_s = 1.0 + max_scale_change_ratio
    return (min_s <= sx <= max_s) and (min_s <= sy <= max_s)


def clip_detections_to_frame(
    detections: sv.Detections,
    frame_w: int,
    frame_h: int,
    min_size: float = 2.0,
) -> sv.Detections:
    """Clip xyxy to frame bounds and drop degenerate boxes."""
    if len(detections) == 0:
        return detections

    xyxy = np.asarray(detections.xyxy, dtype=np.float32).copy()
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, frame_w - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, frame_h - 1)

    ws = xyxy[:, 2] - xyxy[:, 0]
    hs = xyxy[:, 3] - xyxy[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)

    out = sv.Detections(
        xyxy=xyxy[keep],
        confidence=(detections.confidence[keep] if detections.confidence is not None else None),
        class_id=(detections.class_id[keep] if detections.class_id is not None else None),
    )
    if detections.tracker_id is not None:
        out.tracker_id = detections.tracker_id[keep]
    return out
