import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO
from vidstab import VidStab
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch

torch.backends.cudnn.benchmark = True

MODEL_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\train_0413_tune\yolo26n_640\weights\best.pt"
VIDEO_PATH = 0

IMGSZ = 640
CONF_THRES = 0.5
DEVICE = "0"

USE_LEFT_ONLY = True

MIN_BOX_AREA = 1000
MIN_BOX_W = 20
MIN_BOX_H = 20

USE_STABILIZATION = False
SMOOTHING_WINDOW = 30
BORDER_TYPE = "black"
BORDER_SIZE = -20

USE_CLASS_SMOOTHING = True
CLASS_HISTORY_LEN = 5
CENTER_DIST_THRES = 50

USE_PERSISTENCE = True
PERSIST_FRAMES = 0
SHOW_HOLD_LABEL = False

SLICE_HEIGHT = 512
SLICE_WIDTH = 512
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
POSTPROCESS_TYPE = "NMM"
POSTPROCESS_MATCH_METRIC = "IOS"
POSTPROCESS_MATCH_THRESHOLD = 0.5

USE_CENTER_ROI = True
ROI_WIDTH_RATIO = 0.7
ROI_HEIGHT_RATIO = 1.0

track_histories = {}
next_track_id = 0

yolo_model = YOLO(MODEL_PATH)

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRES,
    device=f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu"
)

if USE_STABILIZATION:
    stabilizer = VidStab(
        kp_method="GFTT",
        processing_max_dim=320
    )

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("카메라(또는 영상)를 열 수 없습니다.")

print("실행 중... ESC 누르면 종료됩니다.")
print(f"입력 소스: {VIDEO_PATH}")

src_fps = cap.get(cv2.CAP_PROP_FPS)
if src_fps is None or src_fps <= 0 or src_fps != src_fps:
    src_fps = 20.0

prev_time = time.time()

def get_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def get_box_info(box_row, conf, cls_id):
    x1, y1, x2, y2 = box_row
    cx, cy = get_center(x1, y1, x2, y2)
    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "cx": cx,
        "cy": cy,
        "conf": float(conf),
        "cls_id": int(cls_id)
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    h, w = frame.shape[:2]
    half_w = w // 2

    if USE_LEFT_ONLY:
        frame_one = frame[:, :half_w]
    else:
        frame_one = frame[:, half_w:]

    if USE_STABILIZATION:
        stabilized = stabilizer.stabilize_frame(
            input_frame=frame_one,
            smoothing_window=SMOOTHING_WINDOW,
            border_type=BORDER_TYPE,
            border_size=BORDER_SIZE
        )
        frame_proc = stabilized if stabilized is not None else frame_one
    else:
        frame_proc = frame_one

    if USE_CENTER_ROI:
        infer_frame, roi_x_offset, roi_y_offset, roi_x2, roi_y2 = get_center_roi(
            frame_proc,
            ROI_WIDTH_RATIO,
            ROI_HEIGHT_RATIO
        )
    else:
        infer_frame = frame_proc
        roi_x_offset = 0
        roi_y_offset = 0
        roi_y2, roi_x2 = frame_proc.shape[:2]

    start = time.time()

    result = get_sliced_prediction(
        infer_frame,
        model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        postprocess_type=POSTPROCESS_TYPE,
        postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=POSTPROCESS_MATCH_THRESHOLD,
        verbose=0
    )

    end = time.time()
    infer_fps = 1 / (end - start + 1e-8)

    now = time.time()
    total_fps = 1 / (now - prev_time + 1e-8)
    prev_time = now

    annotated = frame_proc.copy()
    current_dets = []

    for obj in result.object_prediction_list:
        bbox = obj.bbox
        score = obj.score.value
        cls_id = int(obj.category.id)

        x1 = int(bbox.minx) + roi_x_offset
        y1 = int(bbox.miny) + roi_y_offset
        x2 = int(bbox.maxx) + roi_x_offset
        y2 = int(bbox.maxy) + roi_y_offset

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if area < MIN_BOX_AREA:
            continue
        if bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue

        current_dets.append(get_box_info([x1, y1, x2, y2], score, cls_id))

    updated_tracks = {}
    used_prev_ids = set()

    for det in current_dets:
        best_id = None
        best_dist = float("inf")

        for track_id, info in track_histories.items():
            if track_id in used_prev_ids:
                continue

            prev_cx, prev_cy = info["cx"], info["cy"]
            dist = ((det["cx"] - prev_cx) ** 2 + (det["cy"] - prev_cy) ** 2) ** 0.5

            if dist < best_dist and dist < CENTER_DIST_THRES:
                best_dist = dist
                best_id = track_id

        if best_id is not None:
            used_prev_ids.add(best_id)
            prev_info = track_histories[best_id]

            if USE_CLASS_SMOOTHING:
                history = prev_info["history"]
                history.append(det["cls_id"])
                smoothed_cls = Counter(history).most_common(1)[0][0]
            else:
                history = deque([det["cls_id"]], maxlen=CLASS_HISTORY_LEN)
                smoothed_cls = det["cls_id"]

            updated_tracks[best_id] = {
                "cx": det["cx"],
                "cy": det["cy"],
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "conf": det["conf"],
                "cls_id": det["cls_id"],
                "smoothed_cls_id": smoothed_cls,
                "history": history,
                "miss_count": 0,
                "matched": True
            }
        else:
            history = deque([det["cls_id"]], maxlen=CLASS_HISTORY_LEN)

            updated_tracks[next_track_id] = {
                "cx": det["cx"],
                "cy": det["cy"],
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "conf": det["conf"],
                "cls_id": det["cls_id"],
                "smoothed_cls_id": det["cls_id"],
                "history": history,
                "miss_count": 0,
                "matched": True
            }
            next_track_id += 1

    if USE_PERSISTENCE:
        for track_id, info in track_histories.items():
            if track_id in updated_tracks:
                continue

            new_miss = info["miss_count"] + 1

            if new_miss <= PERSIST_FRAMES:
                updated_tracks[track_id] = {
                    "cx": info["cx"],
                    "cy": info["cy"],
                    "x1": info["x1"],
                    "y1": info["y1"],
                    "x2": info["x2"],
                    "y2": info["y2"],
                    "conf": info["conf"],
                    "cls_id": info["cls_id"],
                    "smoothed_cls_id": info["smoothed_cls_id"],
                    "history": info["history"],
                    "miss_count": new_miss,
                    "matched": False
                }

    track_histories = updated_tracks

    if USE_CENTER_ROI:
        cv2.rectangle(
            annotated,
            (roi_x_offset, roi_y_offset),
            (roi_x2, roi_y2),
            (255, 255, 255),
            2
        )

    for track_id, info in track_histories.items():
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
        conf = info["conf"]
        cls_id = info["smoothed_cls_id"]
        cls_name = yolo_model.names[cls_id]

        cls_name_lower = str(cls_name).lower()
        if cls_name_lower == "ripe":
            color = (0, 0, 255)
        elif cls_name_lower == "unripe":
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        if (not info["matched"]) and SHOW_HOLD_LABEL:
            label += " (hold)"

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.putText(
        annotated,
        f"Infer FPS: {infer_fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Total FPS: {total_fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("YOLO + SAHI Real-time Inference", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("종료되었습니다.")
