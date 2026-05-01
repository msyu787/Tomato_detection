"""Tomato observer web UI — Flask on top of `tomato_observer_pipeline`.

Before the HTTP server starts: YOLO is loaded, GPU is warmed with one dummy `predict` (avoids
first-frame stutter). On Start, OpenCV capture is opened in the main thread, then the worker
reads immediately.
"""

from __future__ import annotations

from typing import Any

import csv
import io
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrootutils  # type: ignore[import-untyped]
import torch
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

_REPO_ROOT = Path(__file__).resolve().parent
pyrootutils.setup_root(str(_REPO_ROOT), indicator=".project_root", pythonpath=True)

from src.tracking.pipelines import tomato_observer_pipeline as _top
from src.tracking.pipelines.tomato_observer_pipeline import DEFAULT_CONFIG, run as run_observer

torch.backends.cudnn.benchmark = True

DEFAULT_CONF_THRES = 0.45
DEFAULT_IOU_THRES = 0.50

# Single UI control → full pipeline camera-related settings.
VALID_CAMERA_MODES = frozenset({"webcam", "sbs_left", "sbs_right", "zed"})

_yolo_lock = threading.Lock()
_yolo_cached: YOLO | None = None
_yolo_cache_key: tuple[str, str] | None = None


def ensure_detector_ready(model_path: str, device: str) -> YOLO:
    """Load YOLO once (thread-safe); reused across sessions so Start does not wait on weights."""
    global _yolo_cached, _yolo_cache_key
    resolved = _top._resolve_model_path(str(model_path))
    dev = _top._torch_device_str(device)
    key = (resolved, dev)
    with _yolo_lock:
        if _yolo_cached is not None and _yolo_cache_key == key:
            return _yolo_cached
        m = YOLO(resolved)
        m.to(dev)
        _yolo_cached = m
        _yolo_cache_key = key
        return m


def _prime_yolo_warmup(model: YOLO, device_str: str) -> None:
    """One dummy run so the first real frame is not stuck on CUDA graph compile / kernels."""
    z = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(z, conf=0.5, iou=0.5, verbose=False, imgsz=640)
    if torch.cuda.is_available() and str(device_str).strip().lower() != "cpu":
        torch.cuda.synchronize()


def _camera_mode_label(mode: str) -> str:
    return {
        "webcam": "일반 웹캠",
        "sbs_left": "양안분할·왼쪽",
        "sbs_right": "양안분할·오른쪽",
        "zed": "ZED",
    }.get(mode, mode)


def _pipeline_camera_overrides(mode: str) -> dict[str, Any]:
    """Map UI `camera_mode` to tomato_observer_pipeline fields (no manual tuning in the browser)."""
    if mode == "zed":
        return {
            "camera_backend": "zed",
            "stereo_sbs": False,
            "left_only": False,
            "camera_width": None,
            "camera_height": None,
        }
    if mode == "webcam":
        o: dict[str, Any] = {
            "camera_backend": "opencv",
            "stereo_sbs": False,
            "left_only": False,
            "camera_width": None,
            "camera_height": None,
            "opencv_use_mjpeg": False,
            "opencv_api": "default",
        }
        if sys.platform == "win32":
            o["opencv_api"] = "dshow"
        return o
    if mode == "sbs_left":
        return {
            "camera_backend": "opencv",
            "stereo_sbs": True,
            "left_only": True,
            "camera_width": DEFAULT_CONFIG["camera_width"],
            "camera_height": DEFAULT_CONFIG["camera_height"],
            "opencv_use_mjpeg": True,
            "opencv_api": "default",
        }
    if mode == "sbs_right":
        return {
            "camera_backend": "opencv",
            "stereo_sbs": True,
            "left_only": False,
            "camera_width": DEFAULT_CONFIG["camera_width"],
            "camera_height": DEFAULT_CONFIG["camera_height"],
            "opencv_use_mjpeg": True,
            "opencv_api": "default",
        }
    raise ValueError(f"unsupported camera_mode: {mode}")


app = Flask(__name__)

frame_lock = threading.Lock()
control_lock = threading.Lock()
state_lock = threading.Lock()
stop_event = threading.Event()

worker_thread: threading.Thread | None = None
session_start_time: float | None = None
frame_index = 0
csv_rows: list[dict] = []
pipeline_runtime: dict | None = None

settings_state = {
    "conf_thres": DEFAULT_CONF_THRES,
    "iou_thres": DEFAULT_IOU_THRES,
    "camera_mode": "webcam",
    "opencv_source": 0,
}


def format_elapsed(sec: float) -> str:
    sec = max(int(sec), 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_placeholder(text: str) -> bytes:
    canvas = np.full((720, 1280, 3), 28, dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (360, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (220, 220, 220),
        3,
    )
    ok, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


latest_frame = make_placeholder("Camera stopped")
latest_status: dict = {
    "running": False,
    "connected": False,
    "total_fps": 0.0,
    "ripe_count": 0,
    "unripe_count": 0,
    "total_count": 0,
    "elapsed": "00:00:00",
    "resolution": "-",
    "conf_thres": DEFAULT_CONF_THRES,
    "iou_thres": DEFAULT_IOU_THRES,
    "camera_mode": settings_state.get("camera_mode", "webcam"),
    "camera_label": _camera_mode_label(settings_state.get("camera_mode", "webcam")),
}


def set_idle_state(text: str = "Camera stopped") -> None:
    global latest_frame, latest_status
    with state_lock:
        conf_thres = settings_state["conf_thres"]
        iou_thres = settings_state["iou_thres"]
        cam_mode = settings_state.get("camera_mode", "webcam")

    with frame_lock:
        latest_frame = make_placeholder(text)
        latest_status = {
            "running": False,
            "connected": False,
            "total_fps": 0.0,
            "ripe_count": 0,
            "unripe_count": 0,
            "total_count": 0,
            "elapsed": "00:00:00",
            "resolution": "-",
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "camera_mode": cam_mode,
            "camera_label": _camera_mode_label(cam_mode),
        }


def _session_elapsed_str() -> str:
    global session_start_time
    if session_start_time is None:
        return "00:00:00"
    return format_elapsed(time.time() - session_start_time)


def _on_frame(annotated: np.ndarray, meta: dict) -> None:
    global latest_frame, latest_status, frame_index, csv_rows

    frame_index = int(meta.get("frame_idx", frame_index))

    ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(annotated))
    if not ok:
        return

    elapsed = _session_elapsed_str()
    with frame_lock:
        latest_frame = buffer.tobytes()
        latest_status = {
            "running": True,
            "connected": True,
            "total_fps": round(float(meta.get("fps", 0.0)), 2),
            "ripe_count": int(meta.get("ripe_count", 0)),
            "unripe_count": int(meta.get("unripe_count", 0)),
            "total_count": int(meta.get("total_count", 0)),
            "elapsed": elapsed,
            "resolution": f'{int(meta.get("width", 0))} x {int(meta.get("height", 0))}',
            "conf_thres": settings_state["conf_thres"],
            "iou_thres": settings_state["iou_thres"],
            "camera_mode": settings_state.get("camera_mode", "webcam"),
            "camera_label": _camera_mode_label(settings_state.get("camera_mode", "webcam")),
        }

    rows = meta.get("csv_rows") or []
    with state_lock:
        if rows:
            csv_rows.extend(rows)


def _observer_worker(cfg: dict) -> None:
    stats = cfg.setdefault("_exit_stats", {})
    stats.clear()
    try:
        run_observer(cfg)
    except Exception as e:
        set_idle_state(f"Error: {e}")
    else:
        fi = int(stats.get("frame_idx", 0))
        user_stop = bool(stats.get("user_stop", False))
        if user_stop:
            set_idle_state("Camera stopped")
        elif fi == 0:
            set_idle_state("No frames (wrong camera index, device busy, or try DirectShow / resolution)")
        else:
            set_idle_state("Stream finished")


def start_camera() -> tuple[bool, str]:
    global worker_thread, session_start_time, frame_index, csv_rows, pipeline_runtime, stop_event

    with control_lock:
        if worker_thread is not None and worker_thread.is_alive():
            return False, "already running"

        with state_lock:
            mode = str(settings_state.get("camera_mode", "webcam")).lower().strip()
            if mode not in VALID_CAMERA_MODES:
                return False, f"camera_mode must be one of: {', '.join(sorted(VALID_CAMERA_MODES))}"
            source = int(settings_state.get("opencv_source", 0))
            conf_thres = float(settings_state["conf_thres"])
            iou_thres = float(settings_state["iou_thres"])

        stop_event.clear()
        session_start_time = time.time()
        frame_index = 0
        with state_lock:
            csv_rows = []

        pipeline_runtime = {
            "lock": threading.Lock(),
            "conf": conf_thres,
            "nms_iou": iou_thres,
        }

        cfg: dict = {**DEFAULT_CONFIG}
        cfg.update(
            {
                "model_path": "models/yolo26n_640.pt",
                "device": "0",
                "tracker_type": "bytetrack",
                "source": source,
                "show_window": False,
                "stop_event": stop_event,
                "runtime": pipeline_runtime,
                "on_frame": _on_frame,
                "session_elapsed_fn": _session_elapsed_str,
                "conf": conf_thres,
                "nms_iou": iou_thres,
                "use_stabilization": False,
                "motion_compensation": True,
            }
        )
        cfg.update(_pipeline_camera_overrides(mode))

        cfg["yolo_model"] = ensure_detector_ready(str(cfg["model_path"]), str(cfg["device"]))

        if str(cfg.get("camera_backend", "opencv")).lower() == "opencv":
            try:
                cap = _top._open_source(cfg)
                _top._apply_capture_size(cap, cfg)
                cfg["video_capture"] = cap
            except Exception as e:
                return False, f"카메라를 열 수 없습니다: {e}"

        cfg["_exit_stats"] = {}

        worker_thread = threading.Thread(target=_observer_worker, args=(cfg,), daemon=True)
        worker_thread.start()
        return True, "started"


def stop_camera() -> tuple[bool, str]:
    global worker_thread, session_start_time, pipeline_runtime

    with control_lock:
        stop_event.set()
        thread_ref = worker_thread

    if thread_ref is not None:
        thread_ref.join(timeout=5.0)

    with control_lock:
        worker_thread = None
        session_start_time = None
        pipeline_runtime = None

    set_idle_state("Camera stopped")
    return True, "stopped"


def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("camera_mode")
    legacy = data.get("camera_backend")
    src = data.get("opencv_source")

    with state_lock:
        if mode is not None:
            m = str(mode).lower().strip()
            if m not in VALID_CAMERA_MODES:
                return jsonify({"ok": False, "message": f"camera_mode: one of {sorted(VALID_CAMERA_MODES)}"}), 400
            settings_state["camera_mode"] = m
        elif legacy is not None:
            b = str(legacy).lower().strip()
            settings_state["camera_mode"] = "zed" if b == "zed" else "webcam"
        if src is not None:
            try:
                settings_state["opencv_source"] = int(src)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "message": "opencv_source must be an integer"}), 400

    ok, message = start_camera()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ok, message = stop_camera()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/status")
def api_status():
    with frame_lock:
        return jsonify(dict(latest_status))


@app.route("/api/settings", methods=["POST"])
def api_settings():
    global pipeline_runtime

    data = request.get_json(silent=True) or {}
    try:
        conf_thres = float(data.get("conf_thres", DEFAULT_CONF_THRES))
        iou_thres = float(data.get("iou_thres", DEFAULT_IOU_THRES))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "숫자를 입력해주세요."}), 400

    if not (0 <= conf_thres <= 1):
        return jsonify({"ok": False, "message": "Confidence는 0~1 사이여야 합니다."}), 400
    if not (0 <= iou_thres <= 1):
        return jsonify({"ok": False, "message": "IOU는 0~1 사이여야 합니다."}), 400

    with state_lock:
        settings_state["conf_thres"] = round(conf_thres, 2)
        settings_state["iou_thres"] = round(iou_thres, 2)

    pr = pipeline_runtime
    if pr is not None:
        lock = pr.get("lock")
        if lock is not None:
            with lock:
                pr["conf"] = float(settings_state["conf_thres"])
                pr["nms_iou"] = float(settings_state["iou_thres"])
        else:
            pr["conf"] = float(settings_state["conf_thres"])
            pr["nms_iou"] = float(settings_state["iou_thres"])

    with frame_lock:
        latest_status["conf_thres"] = round(conf_thres, 2)
        latest_status["iou_thres"] = round(iou_thres, 2)

    return jsonify(
        {
            "ok": True,
            "conf_thres": round(conf_thres, 2),
            "iou_thres": round(iou_thres, 2),
        }
    )


@app.route("/download_csv")
def download_csv():
    with state_lock:
        rows = list(csv_rows)

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "timestamp",
            "elapsed",
            "frame_index",
            "ripeness",
            "score",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "area",
            "confidence_threshold",
            "iou_threshold",
            "total_fps",
        ],
    )
    writer.writeheader()
    if rows:
        writer.writerows(rows)

    csv_text = output.getvalue()
    filename = f"tomato_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_text.encode("utf-8-sig"),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    _dev = str(DEFAULT_CONFIG["device"])
    _mp = str(DEFAULT_CONFIG["model_path"])
    print("[tomato_observer_app] detector 로드 중(최초 1회)…", flush=True)
    _m = ensure_detector_ready(_mp, _dev)
    _prime_yolo_warmup(_m, _dev)
    print("[tomato_observer_app] detector 준비됨. 서버 시작…", flush=True)
    set_idle_state("Camera stopped")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False, debug=False)
