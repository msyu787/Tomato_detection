from flask import Flask, Response, jsonify, render_template, request
import threading
import time
import cv2
import torch
import numpy as np
import csv
import io
from datetime import datetime
import pyzed.sl as sl
from ultralytics import YOLO
from vidstab import VidStab
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

torch.backends.cudnn.benchmark = True

MODEL_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\train_0413_tune\yolo26n_640\weights\yolo26n_640.pt"

DEFAULT_CONF_THRES = 0.45
DEFAULT_IOU_THRES = 0.50
DEVICE = "0"

MIN_BOX_AREA = 1000
MIN_BOX_W = 20
MIN_BOX_H = 20

USE_STABILIZATION = True
SMOOTHING_WINDOW = 30
BORDER_TYPE = "black"
BORDER_SIZE = -20

SLICE_HEIGHT = 768
SLICE_WIDTH = 768
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
POSTPROCESS_TYPE = "NMM"
POSTPROCESS_MATCH_METRIC = "IOU"

USE_CENTER_ROI = True
ROI_WIDTH_RATIO = 0.7
ROI_HEIGHT_RATIO = 1.0

app = Flask(__name__)

yolo_model = YOLO(MODEL_PATH)

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=DEFAULT_CONF_THRES,
    device=f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu"
)

stabilizer = None
if USE_STABILIZATION:
    stabilizer = VidStab(
        kp_method="GFTT",
        processing_max_dim=320
    )

frame_lock = threading.Lock()
control_lock = threading.Lock()
state_lock = threading.Lock()
stop_event = threading.Event()

worker_thread = None
session_start_time = None
frame_index = 0
csv_rows = []

settings_state = {
    "conf_thres": DEFAULT_CONF_THRES,
    "iou_thres": DEFAULT_IOU_THRES
}

def make_placeholder(text):
    canvas = np.full((720, 1280, 3), 28, dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (360, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (220, 220, 220),
        3
    )
    ok, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()

latest_frame = make_placeholder("Camera stopped")
latest_status = {
    "running": False,
    "connected": False,
    "total_fps": 0.0,
    "ripe_count": 0,
    "unripe_count": 0,
    "total_count": 0,
    "elapsed": "00:00:00",
    "resolution": "-",
    "conf_thres": DEFAULT_CONF_THRES,
    "iou_thres": DEFAULT_IOU_THRES
}

def format_elapsed(sec):
    sec = max(int(sec), 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def set_idle_state(text="Camera stopped"):
    global latest_frame, latest_status
    with state_lock:
        conf_thres = settings_state["conf_thres"]
        iou_thres = settings_state["iou_thres"]

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
            "iou_thres": iou_thres
        }

def get_center_roi(frame, roi_w_ratio, roi_h_ratio):
    h, w = frame.shape[:2]
    roi_w = int(w * roi_w_ratio)
    roi_h = int(h * roi_h_ratio)
    x1 = max((w - roi_w) // 2, 0)
    y1 = max((h - roi_h) // 2, 0)
    x2 = min(x1 + roi_w, w)
    y2 = min(y1 + roi_h, h)
    roi = frame[y1:y2, x1:x2]
    return roi, x1, y1, x2, y2

def detection_loop():
    global latest_frame, latest_status, frame_index

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        set_idle_state(f"ZED open failed: {err}")
        return

    image = sl.Mat()
    runtime = sl.RuntimeParameters()
    prev_time = time.time()
    last_saved_elapsed_sec = -1

    while not stop_event.is_set():
        grab_status = zed.grab(runtime)
        if grab_status != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01)
            continue

        with state_lock:
            conf_thres = float(settings_state["conf_thres"])
            iou_thres = float(settings_state["iou_thres"])

        model.confidence_threshold = conf_thres

        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if USE_STABILIZATION:
            stabilized = stabilizer.stabilize_frame(
                input_frame=frame,
                smoothing_window=SMOOTHING_WINDOW,
                border_type=BORDER_TYPE,
                border_size=BORDER_SIZE
            )
            frame_proc = stabilized if stabilized is not None else frame
        else:
            frame_proc = frame

        if USE_CENTER_ROI:
            infer_frame, roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi(
                frame_proc,
                ROI_WIDTH_RATIO,
                ROI_HEIGHT_RATIO
            )
        else:
            infer_frame = frame_proc
            roi_x1, roi_y1 = 0, 0
            roi_h, roi_w = frame_proc.shape[:2]
            roi_x2, roi_y2 = roi_w, roi_h

        result = get_sliced_prediction(
            infer_frame,
            model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            postprocess_type=POSTPROCESS_TYPE,
            postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
            postprocess_match_threshold=iou_thres,
            verbose=0
        )

        now = time.time()
        total_fps = 1 / (now - prev_time + 1e-8)
        prev_time = now

        annotated = frame_proc.copy()
        ripe_count = 0
        unripe_count = 0
        total_count = 0
        h, w = annotated.shape[:2]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = format_elapsed(time.time() - session_start_time) if session_start_time else "00:00:00"

        if USE_CENTER_ROI:
            cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)

        frame_rows = []

        for obj in result.object_prediction_list:
            bbox = obj.bbox
            score = float(obj.score.value)
            cls_id = int(obj.category.id)

            x1 = int(bbox.minx) + roi_x1
            y1 = int(bbox.miny) + roi_y1
            x2 = int(bbox.maxx) + roi_x1
            y2 = int(bbox.maxy) + roi_y1

            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if area < MIN_BOX_AREA:
                continue
            if bw < MIN_BOX_W or bh < MIN_BOX_H:
                continue
            if score < conf_thres:
                continue

            cls_name = str(yolo_model.names[cls_id]).lower()

            if cls_name == "ripe":
                color = (0, 0, 255)
                ripe_count += 1
            elif cls_name == "unripe":
                color = (0, 255, 0)
                unripe_count += 1
            else:
                color = (255, 255, 255)

            total_count += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{cls_name} {score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            frame_rows.append({
                "timestamp": timestamp,
                "elapsed": elapsed,
                "frame_index": frame_index,
                "ripeness": cls_name,
                "score": round(score, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": bw,
                "height": bh,
                "area": area,
                "confidence_threshold": conf_thres,
                "iou_threshold": iou_thres,
                "total_fps": round(total_fps, 2)
            })

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        with frame_lock:
            latest_frame = buffer.tobytes()
            latest_status = {
                "running": True,
                "connected": True,
                "total_fps": round(total_fps, 2),
                "ripe_count": ripe_count,
                "unripe_count": unripe_count,
                "total_count": total_count,
                "elapsed": elapsed,
                "resolution": f"{w} x {h}",
                "conf_thres": conf_thres,
                "iou_thres": iou_thres
            }

        with state_lock:
            if frame_rows:
                csv_rows.extend(frame_rows)

        frame_index += 1

    zed.close()
    set_idle_state("Camera stopped")

def start_camera():
    global worker_thread, session_start_time, frame_index, csv_rows
    with control_lock:
        if worker_thread is not None and worker_thread.is_alive():
            return False, "already running"
        stop_event.clear()
        session_start_time = time.time()
        frame_index = 0
        with state_lock:
            csv_rows = []
        worker_thread = threading.Thread(target=detection_loop, daemon=True)
        worker_thread.start()
        return True, "started"

def stop_camera():
    global worker_thread, session_start_time
    with control_lock:
        stop_event.set()
        thread_ref = worker_thread
    if thread_ref is not None:
        thread_ref.join(timeout=2.0)
    with control_lock:
        worker_thread = None
        session_start_time = None
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

    with frame_lock:
        latest_status["conf_thres"] = round(conf_thres, 2)
        latest_status["iou_thres"] = round(iou_thres, 2)

    return jsonify({
        "ok": True,
        "conf_thres": round(conf_thres, 2),
        "iou_thres": round(iou_thres, 2)
    })

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
            "total_fps"
        ]
    )
    writer.writeheader()
    if rows:
        writer.writerows(rows)

    csv_text = output.getvalue()
    filename = f"tomato_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_text.encode("utf-8-sig"),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    set_idle_state("Camera stopped")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
