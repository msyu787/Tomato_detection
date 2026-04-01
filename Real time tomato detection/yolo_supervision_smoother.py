import cv2
import time
from ultralytics import YOLO
from vidstab import VidStab
import torch
import os
import supervision as sv

torch.backends.cudnn.benchmark = True

# =========================
# 경로 / 설정
# =========================
MODEL_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\trained_merge\yolo26n_640\weights\best.pt"

VIDEO_PATH = 0
SAVE_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\result_zed_output_greenbox.mp4"

IMGSZ = 640
CONF_THRES = 0.6
DEVICE = "0"

USE_LEFT_ONLY = True   # True면 왼쪽 화면만 사용, False면 오른쪽 화면 사용

# 작은 bbox 제거 기준
MIN_BOX_AREA = 3000
MIN_BOX_W = 30
MIN_BOX_H = 30

# =========================
# Stabilization 설정
# =========================
USE_STABILIZATION = False
SMOOTHING_WINDOW = 40
BORDER_TYPE = 'black'
BORDER_SIZE = -20

# =========================
# 저장 설정
# =========================
SAVE_VIDEO = True
FRAME_RATE_FALLBACK = 20

# =========================
# 모델 로드
# =========================
model = YOLO(MODEL_PATH)

# =========================
# Stabilizer 초기화
# =========================
if USE_STABILIZATION:
    stabilizer = VidStab(
        kp_method='GFTT',
        processing_max_dim=320
    )

# =========================
# 비디오 캡처
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("카메라(또는 영상)를 열 수 없습니다.")

print("실행 중... ESC 누르면 종료됩니다.")

src_fps = cap.get(cv2.CAP_PROP_FPS)
if src_fps is None or src_fps <= 0 or src_fps != src_fps:
    src_fps = FRAME_RATE_FALLBACK

prev_time = time.time()
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # =========================
    # ZED 좌/우 한쪽만 사용
    # =========================
    h, w = frame.shape[:2]
    half_w = w // 2

    if USE_LEFT_ONLY:
        frame_one = frame[:, :half_w]
    else:
        frame_one = frame[:, half_w:]

    # =========================
    # Video Stabilization
    # =========================
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

    # =========================
    # YOLO 추론
    # =========================
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

    # =========================
    # supervision detections 변환
    # =========================
    detections = sv.Detections.from_ultralytics(results[0])

    # =========================
    # size filtering
    # =========================
    if len(detections) > 0:
        xyxy = detections.xyxy
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        areas = widths * heights

        keep_mask = (
            (areas >= MIN_BOX_AREA) &
            (widths >= MIN_BOX_W) &
            (heights >= MIN_BOX_H)
        )
        detections = detections[keep_mask]

    # =========================
    # 초록색 bbox + 라벨 그리기
    # =========================
    if len(detections) > 0:
        xyxy = detections.xyxy.astype(int)
        class_ids = detections.class_id if detections.class_id is not None else []
        confidences = detections.confidence if detections.confidence is not None else []

        for box, cls_id, conf in zip(xyxy, class_ids, confidences):
            x1, y1, x2, y2 = box
            cls_name = model.names[int(cls_id)]
            label = f"{cls_name} {conf:.2f}"

            color = (0, 255, 0)  # 초록색

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # =========================
    # FPS 표시
    # =========================
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

    # =========================
    # 저장용 VideoWriter 초기화
    # =========================
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

    # =========================
    # 화면 출력
    # =========================
    cv2.imshow("YOLO Real-time (ZED one side)", annotated)

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