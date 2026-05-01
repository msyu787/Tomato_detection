#!/usr/bin/env python3
"""Tomato observer pipeline (YOLO detection + SORT/ByteTrack tracking)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pyrootutils
import torch
import supervision as sv
from trackers import ByteTrackTracker, MotionEstimator, SORTTracker
from ultralytics import YOLO
from trackers.motion.transformation import IdentityTransformation
from vidstab import VidStab

from src.tracking.utils.motion import (
    clip_detections_to_frame,
    is_reasonable_motion_transform,
    warp_detections_abs_to_rel,
    warp_detections_rel_to_abs,
)
from src.tracking.utils.roi import compute_center_roi_by_ratio, crop_frame_with_roi

CLASS_NAMES = {0: "ripe", 1: "unripe"}
REPO_ROOT = Path(pyrootutils.find_root(search_from=__file__, indicator=".project_root"))
DEFAULT_CONFIG = {
    "tracker_type": "bytetrack",
    "source": 0,
    # "opencv": VideoCapture (source = index or path). "zed": Stereolabs ZED left eye (requires pyzed).
    "camera_backend": "opencv",
    # Windows: try "dshow" if MSMF fails with your webcam. "default" uses OpenCV default backend.
    "opencv_api": "default",
    "opencv_use_mjpeg": True,
    "zed_fps": 30,
    # This camera stream is expected to be side-by-side; force stable capture mode.
    "camera_width": 2560,
    "camera_height": 720,
    "model_path": "models/yolo26n_640.pt",
    "device": "0",
    "show_window": True,
    "output_path": None,
    # True: side-by-side stereo — crop to left or right half (see left_only). False: one full image (webcam / ZED / file).
    "stereo_sbs": True,
    "left_only": True,
    "use_center_roi": True,
    "roi_width_ratio": 0.7,
    "roi_height_ratio": 1.0,
    "conf": 0.5,
    "nms_iou": 0.3,
    "min_box_area": 1000,
    "min_box_w": 20,
    "min_box_h": 20,
    "use_stabilization": False,
    "smoothing_window": 30,
    "border_type": "black",
    "border_size": -20,
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.3,
    "minimum_iou_threshold": 0.3,
    "minimum_consecutive_frames": 1,
    "high_conf_det_threshold": 0.25,
    "motion_compensation": True,
    "motion_max_points": 900,
    "motion_min_distance": 6,
    "motion_block_size": 5,
    "motion_quality_level": 0.003,
    "motion_ransac_reproj_threshold": 2.5,
    "show_trace": False,
    "trace_length": 30,
}


def _torch_device_str(device_cfg: object) -> str:
    """Map config device to a string YOLO/torch accepts; fall back to CPU if CUDA is unavailable."""
    raw = str(device_cfg).strip()
    if raw.lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print(
            "[TomatoObserver] CUDA is not available (CPU-only PyTorch or no GPU driver). "
            "Using CPU; set CONFIG['device']='cpu' or install a CUDA build of torch to use GPU."
        )
        return "cpu"
    if raw.lower().startswith("cuda:"):
        return raw
    return f"cuda:{raw}"


def _resolve_model_path(model_path: str) -> str:
    p = REPO_ROOT / model_path
    if p.exists():
        return str(p)
    if Path(model_path).exists():
        return model_path
    candidates = sorted((REPO_ROOT / "models").glob("*.pt")) if (REPO_ROOT / "models").exists() else []
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError(
        f"Model file not found: {model_path}. Put a .pt in 'models/' or set CONFIG['model_path']."
    )


class _ZedCapture:
    """Minimal ZED left-camera reader (optional dependency: Stereolabs pyzed)."""

    def __init__(self, config: dict) -> None:
        try:
            import pyzed.sl as sl
        except ImportError as e:
            raise RuntimeError(
                "camera_backend='zed' requires the ZED SDK Python API (pyzed). "
                "Install from https://www.stereolabs.com/docs/app-development/python/install"
            ) from e
        self._sl = sl
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = int(config.get("zed_fps", 30))
        init.depth_mode = sl.DEPTH_MODE.NONE
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")
        self._image = sl.Mat()
        self._runtime = sl.RuntimeParameters()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.zed.grab(self._runtime) != self._sl.ERROR_CODE.SUCCESS:
            return False, None
        self.zed.retrieve_image(self._image, self._sl.VIEW.LEFT)
        frame = self._image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return True, frame

    def release(self) -> None:
        self.zed.close()

    def get_fps(self) -> float:
        try:
            fps = float(self.zed.get_camera_information().camera_configuration.fps)
            return fps if fps > 0 else 30.0
        except Exception:
            return 30.0

    def get_resolution(self) -> tuple[int, int]:
        try:
            r = self.zed.get_camera_information().camera_configuration.resolution
            return int(r.width), int(r.height)
        except Exception:
            return 1280, 720


def _apply_capture_size(cap: cv2.VideoCapture, config: dict) -> None:
    """Set width/height on a numeric (webcam) source when the config requests it."""
    if isinstance(config["source"], int) or (isinstance(config["source"], str) and str(config["source"]).isdigit()):
        if config.get("camera_width") is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["camera_width"]))
        if config.get("camera_height") is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["camera_height"]))


def _open_source(config: dict) -> cv2.VideoCapture:
    source = config["source"]
    if isinstance(source, int) or (isinstance(source, str) and str(source).isdigit()):
        idx = int(source)
        api = str(config.get("opencv_api", "default")).lower()
        if api == "dshow" and sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        elif api == "msmf" and sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(idx)
        if bool(config.get("opencv_use_mjpeg", True)):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    else:
        cap = cv2.VideoCapture(str(REPO_ROOT / str(source)))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def _make_writer(output_path: Optional[str], width: int, height: int, fps: float):
    if not output_path:
        return None
    out = REPO_ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def _make_tracker(config: dict, fps: float):
    tracker_type = str(config.get("tracker_type", "bytetrack")).lower()
    if tracker_type == "sort":
        return SORTTracker(
            lost_track_buffer=int(config["lost_track_buffer"]),
            frame_rate=float(fps),
            track_activation_threshold=float(config["track_activation_threshold"]),
            minimum_consecutive_frames=int(config.get("minimum_consecutive_frames", 1)),
            minimum_iou_threshold=float(
                config.get("minimum_iou_threshold", config.get("minimum_matching_threshold", 0.3))
            ),
        )
    return ByteTrackTracker(
        lost_track_buffer=int(config["lost_track_buffer"]),
        frame_rate=float(fps),
        track_activation_threshold=float(config["track_activation_threshold"]),
        minimum_consecutive_frames=int(config.get("minimum_consecutive_frames", 1)),
        minimum_iou_threshold=float(config.get("minimum_matching_threshold", 0.3)),
        high_conf_det_threshold=float(config.get("high_conf_det_threshold", 0.25)),
    )


def _yolo_infer(
    yolo: YOLO,
    frame: np.ndarray,
    config: dict,
    *,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
) -> sv.Detections:
    c = float(config["conf"]) if conf is None else float(conf)
    n = float(config["nms_iou"]) if iou is None else float(iou)
    results = yolo.predict(
        source=frame,
        conf=c,
        iou=n,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return sv.Detections.empty()

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)

    keep: list[int] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        bw, bh = float(x2 - x1), float(y2 - y1)
        if bw * bh < int(config["min_box_area"]):
            continue
        if bw < int(config["min_box_w"]) or bh < int(config["min_box_h"]):
            continue
        keep.append(i)

    if not keep:
        return sv.Detections.empty()
    idx = np.asarray(keep, dtype=np.int64)
    return sv.Detections(
        xyxy=xyxy[idx].astype(np.float32),
        confidence=confs[idx].astype(np.float32),
        class_id=classes[idx],
    )


def _tracked_to_csv_rows(
    tracked: sv.Detections,
    *,
    frame_idx: int,
    fps: float,
    conf_thres: float,
    iou_thres: float,
    timestamp: str,
    elapsed: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if tracked.class_id is None or len(tracked) == 0:
        return rows
    tids = tracked.tracker_id
    for i in range(len(tracked)):
        if tids is not None and int(tids[i]) < 0:
            continue
        x1, y1, x2, y2 = (float(tracked.xyxy[i][j]) for j in range(4))
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        bw, bh = xi2 - xi1, yi2 - yi1
        area = bw * bh
        cid = int(tracked.class_id[i])
        cls_key = CLASS_NAMES.get(cid, str(cid))
        rows.append(
            {
                "timestamp": timestamp,
                "elapsed": elapsed,
                "frame_index": frame_idx,
                "ripeness": str(cls_key).lower(),
                "score": round(float(tracked.confidence[i]), 4),
                "x1": xi1,
                "y1": yi1,
                "x2": xi2,
                "y2": yi2,
                "width": bw,
                "height": bh,
                "area": area,
                "confidence_threshold": conf_thres,
                "iou_threshold": iou_thres,
                "total_fps": round(float(fps), 2),
            }
        )
    return rows


def run(config: dict) -> None:
    config = {**DEFAULT_CONFIG, **config}
    backend = str(config.get("camera_backend", "opencv")).lower().strip()
    if backend == "zed":
        config["stereo_sbs"] = False
    stop_event: Optional[threading.Event] = config.get("stop_event")
    on_frame: Optional[Callable[[np.ndarray, dict[str, Any]], None]] = config.get("on_frame")
    runtime: Optional[dict[str, Any]] = config.get("runtime")

    model_path = _resolve_model_path(str(config["model_path"]))
    device_str = _torch_device_str(config["device"])
    prebuilt = config.get("yolo_model")
    if prebuilt is not None:
        yolo_model = prebuilt
    else:
        yolo_model = YOLO(model_path)
    yolo_model.to(device_str)

    zed_cap: Optional[_ZedCapture] = None
    cap: Optional[cv2.VideoCapture] = None

    if backend == "zed":
        zed_cap = _ZedCapture(config)
        fps = zed_cap.get_fps()
        src_w, src_h = zed_cap.get_resolution()
    else:
        cap = config.get("video_capture")
        if cap is None:
            cap = _open_source(config)
        _apply_capture_size(cap, config)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if bool(config.get("stereo_sbs", True)):
        proc_w = src_w // 2 if bool(config["left_only"]) else src_w - (src_w // 2)
    else:
        proc_w = src_w
    proc_h = src_h

    trackers = {cid: _make_tracker(config, fps) for cid in CLASS_NAMES}
    raw_to_stable: Dict[int, Dict[int, int]] = {cid: {} for cid in CLASS_NAMES}
    next_sid: Dict[int, int] = {cid: 1 for cid in CLASS_NAMES}
    seen_ids: Dict[int, set] = {cid: set() for cid in CLASS_NAMES}

    writer = _make_writer(config.get("output_path"), proc_w, proc_h, fps)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator(trace_length=int(config.get("trace_length", 30)))
    show_trace = bool(config.get("show_trace", False))

    prev_time = time.time()
    frame_idx = 0
    tracker_type = str(config.get("tracker_type", "bytetrack")).lower()
    use_stabilization = bool(config.get("use_stabilization", False))
    stabilizer = None
    if use_stabilization:
        stabilizer = VidStab(
            kp_method="GFTT",
            processing_max_dim=320,
        )
    use_motion = bool(config.get("motion_compensation", True))
    motion_est = None
    if use_motion:
        motion_est = MotionEstimator(
            max_points=int(config.get("motion_max_points", 500)),
            min_distance=int(config.get("motion_min_distance", 10)),
            block_size=int(config.get("motion_block_size", 3)),
            quality_level=float(config.get("motion_quality_level", 0.001)),
            ransac_reproj_threshold=float(config.get("motion_ransac_reproj_threshold", 1.0)),
        )
    print(
        f"[TomatoObserver] start | backend={backend} | tracker={tracker_type} | "
        f"stabilization={'on' if use_stabilization else 'off'} | "
        f"motion={'on' if use_motion else 'off'}"
    )

    user_requested_stop = False
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                user_requested_stop = True
                break

            if zed_cap is not None:
                ok, frame = zed_cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok:
                    break

            frame_idx += 1

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)

            infer_conf = float(config["conf"])
            infer_iou = float(config["nms_iou"])
            if runtime is not None:
                lock = runtime.get("lock")
                if lock is not None:
                    with lock:
                        infer_conf = float(runtime.get("conf", infer_conf))
                        infer_iou = float(runtime.get("nms_iou", infer_iou))
                else:
                    infer_conf = float(runtime.get("conf", infer_conf))
                    infer_iou = float(runtime.get("nms_iou", infer_iou))

            h, w = frame.shape[:2]
            if bool(config.get("stereo_sbs", True)):
                half_w = w // 2
                frame_proc = frame[:, :half_w] if bool(config["left_only"]) else frame[:, half_w:]
            else:
                frame_proc = frame
            if use_stabilization and stabilizer is not None:
                stabilized = stabilizer.stabilize_frame(
                    input_frame=frame_proc,
                    smoothing_window=int(config.get("smoothing_window", 30)),
                    border_type=str(config.get("border_type", "black")),
                    border_size=int(config.get("border_size", -20)),
                )
                frame_proc = stabilized if stabilized is not None else frame_proc
            proc_h, proc_w = frame_proc.shape[:2]

            coord = motion_est.update(frame_proc) if motion_est is not None else None
            if coord is not None and not is_reasonable_motion_transform(coord, proc_w, proc_h):
                coord = IdentityTransformation()

            if bool(config["use_center_roi"]):
                roi_x1, roi_y1, roi_x2, roi_y2 = compute_center_roi_by_ratio(
                    frame_proc.shape[:2], float(config["roi_width_ratio"]), float(config["roi_height_ratio"])
                )
                infer_frame = crop_frame_with_roi(frame_proc, (roi_x1, roi_y1, roi_x2, roi_y2))
            else:
                infer_frame = frame_proc
                roi_x1, roi_y1 = 0, 0
                roi_x2, roi_y2 = proc_w, proc_h

            dets = _yolo_infer(yolo_model, infer_frame, config, conf=infer_conf, iou=infer_iou)
            if len(dets) > 0:
                dets.xyxy += np.array([roi_x1, roi_y1, roi_x1, roi_y1], dtype=np.float32)
            if motion_est is not None and coord is not None:
                dets = warp_detections_rel_to_abs(dets, coord)

            boxes_l, confs_l, cls_l, tid_l = [], [], [], []
            for cid, tracker in trackers.items():
                if len(dets) == 0:
                    continue
                cls_dets = dets[dets.class_id == cid]
                if len(cls_dets) == 0:
                    continue

                cls_dets = tracker.update(cls_dets)
                if motion_est is not None and coord is not None:
                    cls_dets = warp_detections_abs_to_rel(cls_dets, coord)
                cls_dets = clip_detections_to_frame(cls_dets, proc_w, proc_h)
                if cls_dets.tracker_id is None or len(cls_dets) == 0:
                    continue

                valid = cls_dets[cls_dets.tracker_id >= 0]
                if len(valid) == 0:
                    continue

                new_idx = [i for i, rid in enumerate(valid.tracker_id) if int(rid) not in raw_to_stable[cid]]
                new_idx.sort(
                    key=lambda i: (
                        0.5 * (valid.xyxy[i][1] + valid.xyxy[i][3]) - 0.5 * (valid.xyxy[i][0] + valid.xyxy[i][2])
                    )
                )
                for i in new_idx:
                    raw_to_stable[cid][int(valid.tracker_id[i])] = next_sid[cid]
                    next_sid[cid] += 1

                valid.tracker_id = np.array([raw_to_stable[cid][int(rid)] for rid in valid.tracker_id], dtype=np.int64)
                seen_ids[cid].update(valid.tracker_id.tolist())
                boxes_l.append(valid.xyxy)
                confs_l.append(valid.confidence)
                cls_l.append(valid.class_id)
                tid_l.append(valid.tracker_id)

            tracked = (
                sv.Detections(
                    xyxy=np.concatenate(boxes_l),
                    confidence=np.concatenate(confs_l),
                    class_id=np.concatenate(cls_l),
                    tracker_id=np.concatenate(tid_l),
                )
                if boxes_l
                else sv.Detections.empty()
            )

            labels = (
                [f"{CLASS_NAMES.get(int(c), int(c))} #{tid}" for tid, c in zip(tracked.tracker_id, tracked.class_id)]
                if tracked.tracker_id is not None
                else []
            )
            annotated = box_annotator.annotate(frame_proc.copy(), detections=tracked)
            annotated = label_annotator.annotate(annotated, detections=tracked, labels=labels)
            if show_trace and tracked.tracker_id is not None:
                annotated = trace_annotator.annotate(annotated, detections=tracked)

            if bool(config["use_center_roi"]):
                cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)

            now = time.time()
            total_fps = 1.0 / max(now - prev_time, 1e-8)
            prev_time = now
            cv2.putText(annotated, f"FPS: {total_fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"ripe {len(seen_ids[0])} / unripe {len(seen_ids[1])}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            if writer is not None:
                writer.write(annotated)
            if bool(config["show_window"]):
                cv2.imshow(f"Tomato Observer ({tracker_type.upper()})", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if on_frame is not None:
                elapsed_s = config.get("session_elapsed_fn")
                elapsed_str = elapsed_s() if callable(elapsed_s) else "00:00:00"
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_part = _tracked_to_csv_rows(
                    tracked,
                    frame_idx=frame_idx,
                    fps=total_fps,
                    conf_thres=infer_conf,
                    iou_thres=infer_iou,
                    timestamp=ts,
                    elapsed=elapsed_str,
                )
                ripe_n = sum(1 for r in csv_part if r["ripeness"] == "ripe")
                unripe_n = sum(1 for r in csv_part if r["ripeness"] == "unripe")
                on_frame(
                    annotated,
                    {
                        "frame_idx": frame_idx,
                        "fps": total_fps,
                        "width": proc_w,
                        "height": proc_h,
                        "csv_rows": csv_part,
                        "ripe_count": ripe_n,
                        "unripe_count": unripe_n,
                        "total_count": len(csv_part),
                    },
                )
    finally:
        exit_stats = config.get("_exit_stats")
        if isinstance(exit_stats, dict):
            exit_stats["frame_idx"] = frame_idx
            exit_stats["user_stop"] = user_requested_stop

        if zed_cap is not None:
            zed_cap.release()
        elif cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if bool(config["show_window"]):
            cv2.destroyAllWindows()

    print(
        f"[TomatoObserver] done | frames={frame_idx} | unique ripe={len(seen_ids[0])}, unripe={len(seen_ids[1])}"
    )
