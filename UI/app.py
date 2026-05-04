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
from flask import Flask, Response, jsonify, render_template, request, send_file
from ultralytics import YOLO

_REPO_ROOT = Path(__file__).resolve().parent
pyrootutils.setup_root(str(_REPO_ROOT), indicator="pyproject.toml", pythonpath=True)

from src.tracking.pipelines import tomato_observer_pipeline as _top
from src.tracking.pipelines.tomato_observer_pipeline import DEFAULT_CONFIG, run as run_observer

torch.backends.cudnn.benchmark = True

DEFAULT_CONF_THRES = 0.65
DEFAULT_IOU_THRES = 0.25

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
    "use_stabilization": True,
    "camera_mode": "zed",
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
    lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines:
        lines = ["Camera stopped", "Press the Start button"]
    y0 = 330
    dy = 56
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (290, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (220, 220, 220),
            3,
        )
    ok, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


latest_frame = make_placeholder("Camera stopped\nPress the Start button")
latest_status: dict = {
    "running": False,
    "connected": False,
    "total_fps": 0.0,
    "ripe_count": 0,
    "unripe_count": 0,
    "ripe_ids": [],
    "unripe_ids": [],
    "current_frame_count": 0,
    "total_count": 0,
    "elapsed": "00:00:00",
    "resolution": "-",
    "conf_thres": DEFAULT_CONF_THRES,
    "iou_thres": DEFAULT_IOU_THRES,
    "use_stabilization": bool(settings_state.get("use_stabilization", True)),
    "camera_mode": settings_state.get("camera_mode", "zed"),
    "camera_label": _camera_mode_label(settings_state.get("camera_mode", "zed")),
}


def set_idle_state(text: str = "Camera stopped\nPress the Start button") -> None:
    global latest_frame, latest_status
    with state_lock:
        conf_thres = settings_state["conf_thres"]
        iou_thres = settings_state["iou_thres"]
        use_stabilization = bool(settings_state.get("use_stabilization", True))
        cam_mode = settings_state.get("camera_mode", "zed")

    with frame_lock:
        latest_frame = make_placeholder(text)
        latest_status = {
            "running": False,
            "connected": False,
            "total_fps": 0.0,
            "ripe_count": 0,
            "unripe_count": 0,
            "ripe_ids": [],
            "unripe_ids": [],
            "current_frame_count": 0,
            "total_count": 0,
            "elapsed": "00:00:00",
            "resolution": "-",
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "use_stabilization": use_stabilization,
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
            "ripe_ids": list(meta.get("ripe_ids", [])),
            "unripe_ids": list(meta.get("unripe_ids", [])),
            "current_frame_count": int(meta.get("current_frame_count", 0)),
            "total_count": int(meta.get("total_count", 0)),
            "elapsed": elapsed,
            "resolution": f'{int(meta.get("width", 0))} x {int(meta.get("height", 0))}',
            "conf_thres": settings_state["conf_thres"],
            "iou_thres": settings_state["iou_thres"],
            "use_stabilization": bool(settings_state.get("use_stabilization", True)),
            "camera_mode": settings_state.get("camera_mode", "zed"),
            "camera_label": _camera_mode_label(settings_state.get("camera_mode", "zed")),
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
            set_idle_state("Camera stopped\nPress the Start button")
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
            mode = str(settings_state.get("camera_mode", "zed")).lower().strip()
            if mode not in VALID_CAMERA_MODES:
                return False, f"camera_mode must be one of: {', '.join(sorted(VALID_CAMERA_MODES))}"
            source = int(settings_state.get("opencv_source", 0))
            conf_thres = float(settings_state["conf_thres"])
            iou_thres = float(settings_state["iou_thres"])
            use_stabilization = bool(settings_state.get("use_stabilization", True))

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
                "model_path": "yolo26n_640.pt",
                "output_path": "recordings/latest_session.mp4",
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
                "use_stabilization": use_stabilization,
                "motion_compensation": True,
            }
        )
        cfg.update(_pipeline_camera_overrides(mode))
        start_message = "started"

        # Fail fast for ZED (or fallback) so API does not report started with zero frames.
        if str(cfg.get("camera_backend", "opencv")).lower() == "zed":
            try:
                zed_probe = _top._ZedCapture(cfg)
                zed_probe.release()
            except Exception as e:
                fallback_source: int | None = None
                for candidate in (2, 0):
                    try:
                        probe_cfg = {**cfg}
                        probe_cfg.update(
                            {
                                "camera_backend": "opencv",
                                "source": candidate,
                                "stereo_sbs": False,
                                "left_only": False,
                                "camera_width": None,
                                "camera_height": None,
                                "opencv_use_mjpeg": False,
                                "opencv_api": "default",
                            }
                        )
                        cap_probe = _top._open_source(probe_cfg)
                        _top._apply_capture_size(cap_probe, probe_cfg)
                        cap_probe.release()
                        cfg.update(probe_cfg)
                        fallback_source = candidate
                        break
                    except Exception:
                        continue

                if fallback_source is None:
                    return False, f"ZED 열기 실패: {e}"

                with state_lock:
                    settings_state["camera_mode"] = "webcam"
                    settings_state["opencv_source"] = fallback_source
                start_message = f"ZED 실패로 webcam({fallback_source}) fallback"

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
        return True, start_message


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

    set_idle_state("Camera stopped\nPress the Start button")
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


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with control_lock:
        running = worker_thread is not None and worker_thread.is_alive()

    if running:
        stop_camera()
        ok, message = start_camera()
        if not ok:
            return jsonify({"ok": False, "message": f"리셋 실패: {message}"}), 500
        return jsonify({"ok": True, "message": "ID 누적값이 초기화되었습니다. (세션 재시작)"})

    with state_lock:
        csv_rows.clear()
    set_idle_state("ID counters reset")
    return jsonify({"ok": True, "message": "ID 누적값이 초기화되었습니다."})


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

    raw_stab = data.get("use_stabilization", settings_state.get("use_stabilization", True))
    if isinstance(raw_stab, bool):
        use_stabilization = raw_stab
    else:
        use_stabilization = str(raw_stab).strip().lower() in {"1", "true", "on", "yes"}

    with state_lock:
        prev_use_stabilization = bool(settings_state.get("use_stabilization", True))
        settings_state["conf_thres"] = round(conf_thres, 2)
        settings_state["iou_thres"] = round(iou_thres, 2)
        settings_state["use_stabilization"] = bool(use_stabilization)

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
        latest_status["use_stabilization"] = bool(use_stabilization)

    restart_applied = False
    if prev_use_stabilization != bool(use_stabilization):
        with control_lock:
            running = worker_thread is not None and worker_thread.is_alive()
        if running:
            stop_camera()
            ok, message = start_camera()
            if not ok:
                return jsonify({"ok": False, "message": f"stabilization 적용 재시작 실패: {message}"}), 500
            restart_applied = True

    return jsonify(
        {
            "ok": True,
            "conf_thres": round(conf_thres, 2),
            "iou_thres": round(iou_thres, 2),
            "use_stabilization": bool(use_stabilization),
            "message": "stabilization 변경 반영을 위해 재시작됨" if restart_applied else "설정 저장됨",
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


@app.route("/download_video")
def download_video():
    video_path = Path(__file__).resolve().parent / "recordings" / "latest_session.mp4"
    if not video_path.exists() or video_path.stat().st_size == 0:
        return jsonify({"ok": False, "message": "다운로드할 영상이 없습니다. 먼저 카메라를 실행해 주세요."}), 404

    filename = f"tomato_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    return send_file(
        str(video_path),
        mimetype="video/mp4",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    _dev = str(DEFAULT_CONFIG["device"])
    _mp = str(DEFAULT_CONFIG["model_path"])
    print("[tomato_observer_app] detector 로드 중(최초 1회)…", flush=True)
    _m = ensure_detector_ready(_mp, _dev)
    _prime_yolo_warmup(_m, _dev)
    print("[tomato_observer_app] detector 준비됨. 서버 시작…", flush=True)
    set_idle_state("Camera stopped\nPress the Start button")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False, debug=False)
