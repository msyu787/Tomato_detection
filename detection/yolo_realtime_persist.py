# After detection persistence
import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO
from vidstab import VidStab
import torch
import os

torch.backends.cudnn.benchmark = True

MODEL_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\trained_merge\yolo26n_640\weights\best.pt"

VIDEO_PATH = 0
SAVE_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\final.mp4"

IMGSZ = 640
CONF_THRES = 0.7
DEVICE = "0"

USE_LEFT_ONLY = True  


MIN_BOX_AREA = 3000
MIN_BOX_W = 30
MIN_BOX_H = 30

# USE_STABILIZATION = True
USE_STABILIZATION = False
SMOOTHING_WINDOW = 30 
BORDER_TYPE = 'black'
BORDER_SIZE = -20

SAVE_VIDEO = True

USE_CLASS_SMOOTHING = True
CLASS_HISTORY_LEN = 5       
CENTER_DIST_THRES = 50  

USE_PERSISTENCE = True
# USE_PERSISTENCE = False
PERSIST_FRAMES =  2  
SHOW_HOLD_LABEL = False    

track_histories = {}
next_track_id = 0

model = YOLO(MODEL_PATH)

if USE_STABILIZATION:
    stabilizer = VidStab(
        kp_method='GFTT',
        processing_max_dim=320
    )

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


if not cap.isOpened():
    raise RuntimeError("카메라(또는 영상)를 열 수 없습니다.")

print("실행 중... ESC 누르면 종료됩니다.")

src_fps = cap.get(cv2.CAP_PROP_FPS)
if src_fps is None or src_fps <= 0 or src_fps != src_fps:
    src_fps = 20.0

prev_time = time.time()
out = None

def get_center(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def get_box_info(box_row, conf, cls_id):
    x1, y1, x2, y2 = box_row
    cx, cy = get_center(x1, y1, x2, y2)
    return {
        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
        "cx": cx, "cy": cy,
        "conf": float(conf),
        "cls_id": int(cls_id)
    }

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

    start = time.time()

    results = model.predict(
        source=frame_proc,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        device=DEVICE,
        half=True,
        verbose=False
    )

    end = time.time()
    infer_fps = 1 / (end - start + 1e-8)

    now = time.time()
    total_fps = 1 / (now - prev_time + 1e-8)
    prev_time = now

    annotated = frame_proc.copy()

    current_dets = []

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)

        for box_row, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = box_row
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if area < MIN_BOX_AREA:
                continue
            if bw < MIN_BOX_W or bh < MIN_BOX_H:
                continue

            current_dets.append(get_box_info(box_row, conf, cls_id))

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

            # class smoothing
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

    for track_id, info in track_histories.items():
        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]
        conf = info["conf"]
        cls_id = info["smoothed_cls_id"]

        cls_name = model.names[cls_id]

        color = (0, 255, 0)       # green

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        # if (not info["matched"]) and SHOW_HOLD_LABEL:
        #     label += " (hold)"

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

    if SAVE_VIDEO and out is None:
        save_h, save_w = annotated.shape[:2]

        save_dir = os.path.dirname(SAVE_PATH)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(SAVE_PATH, fourcc, src_fps, (save_w, save_h))

        if not out.isOpened():
            raise RuntimeError("VideoWriter를 열 수 없습니다. SAVE_PATH 또는 코덱을 확인하세요.")

        print(f"저장 시작: {SAVE_PATH}")
        print(f"저장 FPS: {src_fps}, 저장 크기: ({save_w}, {save_h})")

    if SAVE_VIDEO and out is not None:
        out.write(annotated)

    cv2.imshow("YOLO Real-time (ZED one side + persistence)", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()

if out is not None:
    out.release()

cv2.destroyAllWindows()

print("종료되었습니다.")
if SAVE_VIDEO:
    print(f"저장 완료: {SAVE_PATH}")
