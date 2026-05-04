#!/usr/bin/env python3
"""
YOLO + ByteTrack 기본 트래킹

동영상 파일 · 웹캠 모두 지원

트래커·카메라 모션 보정: Roboflow trackers — ByteTrackTracker + MotionEstimator
(CMC, Trackers 2.2+). 검출은 보정 좌표계에서 연관되고, 출력은 다시 현재 프레임 좌표로 되돌립니다.

실행 시 ``run(config)`` 에 넘기는 dict 는 ``scripts/trackers/basic_bytetracker.py`` 에서 정의합니다.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrootutils
import supervision as sv
from trackers import ByteTrackTracker, MotionEstimator
from trackers.motion.transformation import IdentityTransformation
from ultralytics import YOLO

from src.tracking.utils.motion import (
    clip_detections_to_frame,
    is_reasonable_motion_transform,
    warp_detections_abs_to_rel,
    warp_detections_rel_to_abs,
)
from src.tracking.utils.roi import compute_roi, yolo_detections_with_roi

REPO_ROOT = Path(pyrootutils.find_root(search_from=__file__, indicator=".project_root"))

CLASS_NAMES = {0: "ripe", 1: "unripe"}


def get_model(model_path: str) -> YOLO:
    p = REPO_ROOT / model_path
    if p.exists():
        return YOLO(str(p))
    if Path(model_path).exists():
        return YOLO(model_path)
    candidates = sorted((REPO_ROOT / "models").glob("*.pt")) if (REPO_ROOT / "models").exists() else []
    if candidates:
        return YOLO(str(candidates[-1]))
    return YOLO("yolov8n.pt")


def open_source(source: str) -> cv2.VideoCapture:
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(REPO_ROOT / source))
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
    return cap


def make_writer(output_path: Optional[str], width: int, height: int, fps: float):
    if not output_path:
        return None
    out = REPO_ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def make_tracker(config: dict, fps: float) -> ByteTrackTracker:
    iou_thr = float(
        config.get("minimum_iou_threshold", config.get("minimum_matching_threshold", 0.8))
    )
    return ByteTrackTracker(
        lost_track_buffer=int(config["lost_track_buffer"]),
        frame_rate=float(fps),
        track_activation_threshold=float(config["track_activation_threshold"]),
        minimum_consecutive_frames=int(config.get("minimum_consecutive_frames", 1)),
        minimum_iou_threshold=iou_thr,
        high_conf_det_threshold=float(config.get("high_conf_det_threshold", 0.25)),
    )


def run(config: dict) -> dict:
    """ByteTrack 트래킹 실행.

    Returns:
        dict:
            mot_rows    (List[tuple]): (frame_id, track_id, x, y, w, h, conf, class_id)
            fps_avg     (float)
            total_frames (int)
            unique_ids  (Dict[int, set]): {class_id: set of track_ids}
    """
    model = get_model(config["model_path"])
    cap   = open_source(config["source"])
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi   = compute_roi(w, h, config.get("roi_half_width"))
    if roi is not None:
        print(f"[ByteTrack] ROI x=[{roi[0]}, {roi[2]}] (roi_half_width={config.get('roi_half_width')})")

    use_motion = bool(config.get("motion_compensation", True))
    motion_est: Optional[MotionEstimator] = None
    if use_motion:
        motion_est = MotionEstimator(
            max_points=int(config.get("motion_max_points", 500)),
            min_distance=int(config.get("motion_min_distance", 10)),
            block_size=int(config.get("motion_block_size", 3)),
            quality_level=float(config.get("motion_quality_level", 0.001)),
            ransac_reproj_threshold=float(config.get("motion_ransac_reproj_threshold", 1.0)),
        )
        print("[ByteTrack] motion: trackers.MotionEstimator + trackers.ByteTrackTracker (on)")
    else:
        print("[ByteTrack] motion: off")

    trackers:      Dict[int, ByteTrackTracker]  = {cid: make_tracker(config, fps) for cid in CLASS_NAMES}
    # raw tracker ID → stable ID (클래스별, cy-cx 오름차순으로 발급)
    raw_to_stable: Dict[int, Dict[int, int]]  = {cid: {} for cid in CLASS_NAMES}
    next_sid:      Dict[int, int]             = {cid: 1 for cid in CLASS_NAMES}
    seen_ids:      Dict[int, set]             = {cid: set() for cid in CLASS_NAMES}
    mot_rows:      List[Tuple]                = []

    writer        = make_writer(config.get("output_path"), w, h, fps)
    show_window   = config.get("show_window", False)
    show_trace    = config.get("show_trace", False)

    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(trace_length=config.get("trace_length", 30))

    frame_idx = 0
    fps_acc   = 0.0

    import time
    print(f"[ByteTrack] 시작...")
    bad_motion_warns = 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        coord = motion_est.update(frame) if motion_est is not None else None
        if coord is not None and not is_reasonable_motion_transform(coord, w, h):
            if bad_motion_warns < 5:
                print("[ByteTrack][WARN] unstable motion transform -> fallback to identity")
                bad_motion_warns += 1
            coord = IdentityTransformation()

        all_dets = yolo_detections_with_roi(
            model, frame, config["conf"], config["iou"], roi,
        )
        if motion_est is not None and coord is not None:
            all_dets = warp_detections_rel_to_abs(all_dets, coord)

        boxes_l, confs_l, cls_l, tid_l = [], [], [], []

        for cid, tracker in trackers.items():
            mask     = all_dets.class_id == cid
            cls_dets = all_dets[mask]
            if len(cls_dets) == 0:
                continue
            cls_dets = tracker.update(cls_dets)
            if motion_est is not None and coord is not None:
                cls_dets = warp_detections_abs_to_rel(cls_dets, coord)
            cls_dets = clip_detections_to_frame(cls_dets, w, h)
            if cls_dets.tracker_id is None or len(cls_dets) == 0:
                continue
            work = cls_dets[cls_dets.tracker_id >= 0]
            if len(work) == 0:
                continue
            # tracker.py 방식: 신규 raw ID를 cy-cx 오름차순(우상단→좌하단)으로 정렬 후 순번 발급
            new_idx = [i for i, r in enumerate(work.tracker_id)
                      if int(r) not in raw_to_stable[cid]]
            new_idx.sort(key=lambda i: (
                0.5 * (work.xyxy[i][1] + work.xyxy[i][3])   # cy
                - 0.5 * (work.xyxy[i][0] + work.xyxy[i][2])  # cx
            ))
            for i in new_idx:
                raw_to_stable[cid][int(work.tracker_id[i])] = next_sid[cid]
                next_sid[cid] += 1
            work.tracker_id = np.array(
                [raw_to_stable[cid][int(r)] for r in work.tracker_id], dtype=np.int64
            )
            seen_ids[cid].update(work.tracker_id.tolist())
            boxes_l.append(work.xyxy)
            confs_l.append(work.confidence)
            cls_l.append(work.class_id)
            tid_l.append(work.tracker_id)

        if boxes_l:
            detections = sv.Detections(
                xyxy=np.concatenate(boxes_l),
                confidence=np.concatenate(confs_l),
                class_id=np.concatenate(cls_l),
                tracker_id=np.concatenate(tid_l),
            )
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                mot_rows.append((
                    frame_idx, int(detections.tracker_id[i]),
                    x1, y1, x2 - x1, y2 - y1,
                    float(detections.confidence[i]), int(detections.class_id[i]),
                ))
        else:
            detections = sv.Detections.empty()

        elapsed = time.perf_counter() - t0
        fps_acc = 0.1 * (1 / elapsed if elapsed > 0 else 0) + 0.9 * fps_acc if fps_acc else (1 / elapsed if elapsed > 0 else 0)

        if writer or show_window:
            labels = [
                f"{CLASS_NAMES.get(int(c), int(c))} #{tid}"
                for tid, c in zip(detections.tracker_id, detections.class_id)
            ] if detections.tracker_id is not None else []

            annotated = box_annotator.annotate(frame.copy(), detections=detections)
            annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
            if show_trace and detections.tracker_id is not None:
                annotated = trace_annotator.annotate(annotated, detections=detections)

            if roi is not None:
                cv2.rectangle(
                    annotated, (roi[0], roi[1]), (roi[2], roi[3]), (255, 200, 0), 2,
                )

            for i, text in enumerate([
                f"Frame {frame_idx}",
                f"ripe   total: {len(seen_ids[0])}",
                f"unripe total: {len(seen_ids[1])}",
            ]):
                y = 30 + i * 28
                cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
                cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

            if writer:
                writer.write(annotated)
            if show_window:
                cv2.imshow("YOLO + ByteTrack", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"[ByteTrack] 완료 | {frame_idx}프레임 | FPS={fps_acc:.1f} | "
          f"ripe={len(seen_ids[0])} unripe={len(seen_ids[1])}")

    return {
        "mot_rows":     mot_rows,
        "fps_avg":      fps_acc,
        "total_frames": frame_idx,
        "unique_ids":   seen_ids,
    }
