import os
import re
import json
import cv2
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO
from vidstab import VidStab
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# MODEL_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\trained_merge\yolo26n_640\weights\best.pt "
VIDEO_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\0414.mp4"
COCO_JSON_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\metric_coco\15fps\_annotations.coco.json"

OUTPUT_DIR = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\15fps\result1"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "summary.txt")

IMGSZ = 768
CONF_THRES = 0.4
DEVICE = "0"

USE_ZED_ONE_SIDE = False
USE_LEFT_ONLY = False

USE_SAHI = False
USE_CENTER_ROI = True
USE_STABILIZATION = False
USE_CLASS_SMOOTHING = False
USE_PERSISTENCE = False
CROP_GT_TO_ROI = True

SHOW_WINDOW = True
SAVE_IMAGE = True
SAVE_SUMMARY_TXT = True
PRINT_PER_FRAME_LOG = True

DRAW_GT = True
DRAW_PRED = True
DRAW_FRAME_METRICS = True
SHOW_HOLD_LABEL = False

MIN_BOX_AREA = 2000
MIN_BOX_W = 50
MIN_BOX_H = 50

IOU_THRESH = 0.25
REQUIRE_CLASS_MATCH = True

SLICE_HEIGHT = 640
SLICE_WIDTH = 640

OVERLAP_HEIGHT_RATIO = 0.25
OVERLAP_WIDTH_RATIO = 0.25
POSTPROCESS_TYPE = "GREEDYNMM" # "NMM": Non-Maximum Merging, 합치는거
POSTPROCESS_MATCH_METRIC = "IOS" #  "IOS": Intersection over Smaller, 두 박스의 겹친 면적을 더 작은 박스의 면적으로 나눈 값
POSTPROCESS_MATCH_THRESHOLD = 0.5

ROI_WIDTH_RATIO = 0.3
ROI_HEIGHT_RATIO = 0.6

SMOOTHING_WINDOW = 30
BORDER_TYPE = "black"
BORDER_SIZE = -20

CLASS_HISTORY_LEN = 5
CENTER_DIST_THRES = 50

PERSIST_FRAMES = 1

COMPUTE_MAP50 = True
COMPUTE_MAP50_95 = True
MAP50_IOU_THRESHOLDS = [0.50]
MAP50_95_IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

track_histories = {}
next_track_id = 0

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_frame_index_from_name(name):
    m = re.search(r"_frame_(\d+)", name)
    if m:
        return int(m.group(1))
    return None

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    x, y, w, h = map(float, [x, y, w, h])
    return [x, y, x + w, y + h]

def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h

def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_area_xyxy(box_a)
    area_b = box_area_xyxy(box_b)
    union = area_a + area_b - inter_area + 1e-9

    return inter_area / union

def safe_div(a, b):
    return a / (b + 1e-9)

def compute_prf1(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def normalize_name(name):
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

def build_gt_to_yolo_class_map(cat_id_to_name, model_names):
    gt_to_yolo = {}

    normalized_model = {
        normalize_name(v): int(k)
        for k, v in model_names.items()
    }

    for cat_id, cat_name in cat_id_to_name.items():
        key = normalize_name(cat_name)
        if key in normalized_model:
            gt_to_yolo[cat_id] = normalized_model[key]

    return gt_to_yolo

def get_class_name(model_names, cls_id):
    if isinstance(model_names, dict):
        return str(model_names.get(int(cls_id), str(cls_id)))
    if isinstance(model_names, list):
        if 0 <= int(cls_id) < len(model_names):
            return str(model_names[int(cls_id)])
    return str(cls_id)

def get_eval_class_ids(gt_to_yolo_cls):
    return sorted(set(int(v) for v in gt_to_yolo_cls.values()))

def preprocess_frame(frame):
    if USE_ZED_ONE_SIDE:
        h, w = frame.shape[:2]
        half_w = w // 2
        if USE_LEFT_ONLY:
            frame = frame[:, :half_w]
        else:
            frame = frame[:, half_w:]
    return frame

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

def load_coco_gt(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco["categories"]
    images = coco["images"]
    annotations = coco["annotations"]

    cat_id_to_name = {int(c["id"]): c["name"] for c in categories}

    image_map = {}
    for img in images:
        original_name = img.get("extra", {}).get("name", img["file_name"])
        frame_idx = parse_frame_index_from_name(original_name)

        image_map[img["id"]] = {
            "file_name": img["file_name"],
            "original_name": original_name,
            "frame_idx": frame_idx,
            "width": img["width"],
            "height": img["height"],
        }

    gt_by_frame = {}
    for ann in annotations:
        image_id = ann["image_id"]
        img_info = image_map[image_id]
        frame_idx = img_info["frame_idx"]

        if frame_idx is None:
            continue

        gt_item = {
            "ann_id": ann["id"],
            "image_id": image_id,
            "frame_idx": frame_idx,
            "original_name": img_info["original_name"],
            "file_name": img_info["file_name"],
            "category_id": int(ann["category_id"]),
            "class_name": cat_id_to_name.get(int(ann["category_id"]), str(ann["category_id"])),
            "bbox": xywh_to_xyxy(ann["bbox"]),
            "iscrowd": ann.get("iscrowd", 0),
        }

        gt_by_frame.setdefault(frame_idx, []).append(gt_item)

    return gt_by_frame, cat_id_to_name

def crop_gt_to_roi(gt_list, roi_info):
    if not CROP_GT_TO_ROI:
        return gt_list

    if roi_info is None or not roi_info["enabled"]:
        return gt_list

    rx1 = float(roi_info["x1"])
    ry1 = float(roi_info["y1"])
    rx2 = float(roi_info["x2"])
    ry2 = float(roi_info["y2"])

    cropped_gt = []

    for gt in gt_list:
        x1, y1, x2, y2 = map(float, gt["bbox"])

        nx1 = max(x1, rx1)
        ny1 = max(y1, ry1)
        nx2 = min(x2, rx2)
        ny2 = min(y2, ry2)

        if nx2 <= nx1 or ny2 <= ny1:
            continue

        new_gt = gt.copy()
        new_gt["bbox"] = [nx1, ny1, nx2, ny2]
        cropped_gt.append(new_gt)

    return cropped_gt

def filter_pred_boxes(pred_list):
    filtered = []

    for pred in pred_list:
        x1, y1, x2, y2 = map(float, pred["bbox"])
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if area < MIN_BOX_AREA:
            continue
        if bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue

        filtered.append({
            "bbox": [x1, y1, x2, y2],
            "conf": float(pred["conf"]),
            "cls_id": int(pred["cls_id"]),
        })

    return filtered

def predict_on_frame_yolo(model, frame):
    results = model.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        device=DEVICE,
        half=True,
        verbose=False
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)

    pred_list = []
    for box, conf, cls_id in zip(xyxy, confs, clss):
        pred_list.append({
            "bbox": list(map(float, box)),
            "conf": float(conf),
            "cls_id": int(cls_id)
        })

    return filter_pred_boxes(pred_list)

def predict_on_frame_sahi(detection_model, frame):
    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        postprocess_type=POSTPROCESS_TYPE,
        postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=POSTPROCESS_MATCH_THRESHOLD,
        perform_standard_pred=False,
        verbose=0
    )

    pred_list = []
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        pred_list.append({
            "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
            "conf": float(obj.score.value),
            "cls_id": int(obj.category.id)
        })

    return filter_pred_boxes(pred_list)

def run_detector(yolo_model, sahi_model, frame):
    if USE_CENTER_ROI:
        infer_frame, roi_x_offset, roi_y_offset, roi_x2, roi_y2 = get_center_roi(
            frame,
            ROI_WIDTH_RATIO,
            ROI_HEIGHT_RATIO
        )
    else:
        infer_frame = frame
        roi_x_offset = 0
        roi_y_offset = 0
        h, w = frame.shape[:2]
        roi_x2 = w
        roi_y2 = h

    if USE_SAHI:
        pred_list = predict_on_frame_sahi(sahi_model, infer_frame)
    else:
        pred_list = predict_on_frame_yolo(yolo_model, infer_frame)

    adjusted = []
    for pred in pred_list:
        x1, y1, x2, y2 = pred["bbox"]
        adjusted.append({
            "bbox": [
                x1 + roi_x_offset,
                y1 + roi_y_offset,
                x2 + roi_x_offset,
                y2 + roi_y_offset
            ],
            "conf": pred["conf"],
            "cls_id": pred["cls_id"]
        })

    roi_info = {
        "enabled": USE_CENTER_ROI,
        "x1": roi_x_offset,
        "y1": roi_y_offset,
        "x2": roi_x2,
        "y2": roi_y2
    }

    return adjusted, roi_info

def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def apply_tracking_smoothing(pred_list):
    global track_histories, next_track_id

    current_dets = []
    for pred in pred_list:
        x1, y1, x2, y2 = pred["bbox"]
        cx, cy = get_center(pred["bbox"])
        current_dets.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy,
            "conf": pred["conf"],
            "cls_id": pred["cls_id"]
        })

    updated_tracks = {}
    used_prev_ids = set()

    for det in current_dets:
        best_id = None
        best_dist = float("inf")

        for track_id, info in track_histories.items():
            if track_id in used_prev_ids:
                continue

            dist = ((det["cx"] - info["cx"]) ** 2 + (det["cy"] - info["cy"]) ** 2) ** 0.5
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

    final_preds = []
    for track_id, info in track_histories.items():
        final_preds.append({
            "bbox": [info["x1"], info["y1"], info["x2"], info["y2"]],
            "conf": info["conf"],
            "cls_id": info["smoothed_cls_id"],
            "matched": info["matched"]
        })

    return final_preds

def greedy_match(gt_list, pred_list, gt_to_yolo_cls=None):
    candidates = []

    for gi, gt in enumerate(gt_list):
        for pi, pred in enumerate(pred_list):
            iou = compute_iou(gt["bbox"], pred["bbox"])

            if iou < IOU_THRESH:
                continue

            if REQUIRE_CLASS_MATCH:
                if gt_to_yolo_cls is None:
                    continue

                gt_yolo_cls = gt_to_yolo_cls.get(gt["category_id"], None)
                if gt_yolo_cls is None:
                    continue

                if gt_yolo_cls != pred["cls_id"]:
                    continue

            candidates.append((iou, gi, pi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)

        matches.append({
            "gt_index": gi,
            "pred_index": pi,
            "iou": iou,
            "gt": gt_list[gi],
            "pred": pred_list[pi]
        })

    unmatched_gt = [gt_list[i] for i in range(len(gt_list)) if i not in matched_gt]
    unmatched_pred = [pred_list[i] for i in range(len(pred_list)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred

def draw_boxes(frame, gt_list, pred_list, model_names, roi_info=None):
    vis = frame.copy()

    if roi_info is not None and roi_info["enabled"]:
        cv2.rectangle(
            vis,
            (int(roi_info["x1"]), int(roi_info["y1"])),
            (int(roi_info["x2"]), int(roi_info["y2"])),
            (255, 255, 255),
            2
        )

    if DRAW_GT:
        for gt in gt_list:
            x1, y1, x2, y2 = map(int, gt["bbox"])
            cls_name = gt["class_name"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                vis,
                f"GT: {cls_name}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    if DRAW_PRED:
        for pred in pred_list:
            x1, y1, x2, y2 = map(int, pred["bbox"])
            cls_id = pred["cls_id"]
            conf = pred["conf"]
            cls_name = get_class_name(model_names, cls_id)

            color = (255, 255, 255)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"Pred: {cls_name} {conf:.2f}"
            if ("matched" in pred) and (not pred["matched"]) and SHOW_HOLD_LABEL:
                label += " (hold)"

            cv2.putText(
                vis,
                label,
                (x1, min(y2 + 25, vis.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2
            )

    return vis

def init_per_class_stats(class_ids):
    stats = {}
    for cls_id in class_ids:
        stats[int(cls_id)] = {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    return stats

def update_per_class_stats(per_class_stats, gt_list, pred_list, matches, gt_to_yolo_cls, class_ids):
    class_ids_set = set(class_ids)

    gt_count_by_cls = Counter()
    for gt in gt_list:
        gt_cls = gt_to_yolo_cls.get(gt["category_id"], None)
        if gt_cls is not None and gt_cls in class_ids_set:
            gt_count_by_cls[int(gt_cls)] += 1

    pred_count_by_cls = Counter()
    for pred in pred_list:
        pred_cls = int(pred["cls_id"])
        if pred_cls in class_ids_set:
            pred_count_by_cls[pred_cls] += 1

    tp_count_by_cls = Counter()
    for match in matches:
        pred_cls = int(match["pred"]["cls_id"])
        if pred_cls in class_ids_set:
            tp_count_by_cls[pred_cls] += 1

    for cls_id in class_ids:
        cls_id = int(cls_id)
        tp = tp_count_by_cls.get(cls_id, 0)
        fp = pred_count_by_cls.get(cls_id, 0) - tp
        fn = gt_count_by_cls.get(cls_id, 0) - tp

        per_class_stats[cls_id]["tp"] += tp
        per_class_stats[cls_id]["fp"] += fp
        per_class_stats[cls_id]["fn"] += fn

def init_ap_buffers(class_ids):
    gt_buffer = {}
    pred_buffer = {}

    for cls_id in class_ids:
        gt_buffer[int(cls_id)] = {}
        pred_buffer[int(cls_id)] = []

    return gt_buffer, pred_buffer

def update_ap_buffers(gt_buffer, pred_buffer, frame_idx, gt_list, pred_list, gt_to_yolo_cls, class_ids):
    class_ids_set = set(class_ids)

    for gt in gt_list:
        gt_cls = gt_to_yolo_cls.get(gt["category_id"], None)
        if gt_cls is None:
            continue
        gt_cls = int(gt_cls)
        if gt_cls not in class_ids_set:
            continue

        gt_buffer[gt_cls].setdefault(frame_idx, [])
        gt_buffer[gt_cls][frame_idx].append({
            "bbox": list(map(float, gt["bbox"]))
        })

    for pred in pred_list:
        pred_cls = int(pred["cls_id"])
        if pred_cls not in class_ids_set:
            continue

        pred_buffer[pred_cls].append({
            "frame_idx": int(frame_idx),
            "bbox": list(map(float, pred["bbox"])),
            "conf": float(pred["conf"])
        })

def compute_ap_from_pr(recalls, precisions):
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], np.asarray(recalls), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(precisions), [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return float(ap)

def compute_class_ap(gt_frames, pred_list, iou_thresh):
    npos = sum(len(v) for v in gt_frames.values())

    if npos == 0:
        return np.nan

    preds = sorted(pred_list, key=lambda x: x["conf"], reverse=True)

    used = {}
    for frame_idx, gts in gt_frames.items():
        used[frame_idx] = [False] * len(gts)

    tp_flags = []
    fp_flags = []

    for pred in preds:
        frame_idx = pred["frame_idx"]
        gts = gt_frames.get(frame_idx, [])

        best_iou = -1.0
        best_gt_idx = -1

        for gi, gt in enumerate(gts):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_gt_idx >= 0 and best_iou >= iou_thresh and not used[frame_idx][best_gt_idx]:
            used[frame_idx][best_gt_idx] = True
            tp_flags.append(1.0)
            fp_flags.append(0.0)
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)

    if len(tp_flags) == 0:
        return 0.0

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)

    recalls = tp_cum / (npos + 1e-9)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = compute_ap_from_pr(recalls, precisions)
    return float(ap)

def compute_map_results(gt_buffer, pred_buffer, class_ids, iou_thresholds):
    per_class_result = {}
    valid_class_scores = []

    for cls_id in class_ids:
        cls_id = int(cls_id)
        ap_list = []

        for thr in iou_thresholds:
            ap = compute_class_ap(
                gt_frames=gt_buffer[cls_id],
                pred_list=pred_buffer[cls_id],
                iou_thresh=thr
            )
            ap_list.append(ap)

        valid_aps = [x for x in ap_list if not np.isnan(x)]
        mean_ap = float(np.mean(valid_aps)) if len(valid_aps) > 0 else np.nan

        per_class_result[cls_id] = {
            "aps": {float(thr): float(ap_list[i]) for i, thr in enumerate(iou_thresholds)},
            "mean_ap": mean_ap
        }

        if not np.isnan(mean_ap):
            valid_class_scores.append(mean_ap)

    map_value = float(np.mean(valid_class_scores)) if len(valid_class_scores) > 0 else np.nan
    return per_class_result, map_value

def format_float(x):
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    return f"{x:.6f}"

def main():
    global track_histories, next_track_id

    ensure_dir(OUTPUT_DIR)
    if SAVE_IMAGE:
        ensure_dir(VIS_DIR)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"VIDEO_PATH not found: {VIDEO_PATH}")
    if not os.path.exists(COCO_JSON_PATH):
        raise FileNotFoundError(f"COCO_JSON_PATH not found: {COCO_JSON_PATH}")

    print("Loading GT...")
    gt_by_frame, cat_id_to_name = load_coco_gt(COCO_JSON_PATH)
    eval_frame_indices = sorted(gt_by_frame.keys())
    print(f"Number of evaluation frames: {len(eval_frame_indices)}")

    print("Loading model...")
    yolo_model = YOLO(MODEL_PATH)
    model_names = yolo_model.names

    sahi_model = None
    if USE_SAHI:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=MODEL_PATH,
            confidence_threshold=CONF_THRES,
            device=f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu",
            image_size=IMGSZ
        )

    stabilizer = None
    if USE_STABILIZATION:
        stabilizer = VidStab(
            kp_method="GFTT",
            processing_max_dim=320
        )

    print("YOLO class names:", model_names)
    print("COCO category map:", cat_id_to_name)

    gt_to_yolo_cls = build_gt_to_yolo_class_map(cat_id_to_name, model_names)
    print("GT category_id -> YOLO cls_id:", gt_to_yolo_cls)

    class_ids = get_eval_class_ids(gt_to_yolo_cls)
    class_ids_set = set(class_ids)

    print("Evaluation class ids:", class_ids)
    print("Evaluation class names:", [get_class_name(model_names, c) for c in class_ids])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_eval = 0
    total_class_correct_pred = 0

    per_class_stats = init_per_class_stats(class_ids)
    ap_gt_buffer, ap_pred_buffer = init_ap_buffers(class_ids)

    track_histories = {}
    next_track_id = 0

    for frame_idx in eval_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"[Warning] Failed to read frame {frame_idx}")
            continue

        frame = preprocess_frame(frame)

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

        raw_gt_list = gt_by_frame[frame_idx]
        pred_list, roi_info = run_detector(yolo_model, sahi_model, frame_proc)

        if USE_CLASS_SMOOTHING or USE_PERSISTENCE:
            pred_list = apply_tracking_smoothing(pred_list)

        gt_list = crop_gt_to_roi(raw_gt_list, roi_info)

        matches, unmatched_gt, unmatched_pred = greedy_match(
            gt_list,
            pred_list,
            gt_to_yolo_cls=gt_to_yolo_cls
        )

        tp = len(matches)
        fn = len(unmatched_gt)
        fp = len(unmatched_pred)

        frame_pred_eval = sum(
            1 for pred in pred_list
            if int(pred["cls_id"]) in class_ids_set
        )
        frame_class_correct = len(matches)

        total_tp += tp
        total_fn += fn
        total_fp += fp
        total_pred_eval += frame_pred_eval
        total_class_correct_pred += frame_class_correct

        update_per_class_stats(
            per_class_stats=per_class_stats,
            gt_list=gt_list,
            pred_list=pred_list,
            matches=matches,
            gt_to_yolo_cls=gt_to_yolo_cls,
            class_ids=class_ids
        )

        update_ap_buffers(
            gt_buffer=ap_gt_buffer,
            pred_buffer=ap_pred_buffer,
            frame_idx=frame_idx,
            gt_list=gt_list,
            pred_list=pred_list,
            gt_to_yolo_cls=gt_to_yolo_cls,
            class_ids=class_ids
        )

        frame_precision, frame_recall, frame_f1 = compute_prf1(tp, fp, fn)

        vis = draw_boxes(frame_proc, gt_list, pred_list, model_names, roi_info=roi_info)

        if DRAW_FRAME_METRICS:
            cv2.putText(
                vis,
                f"frame: {frame_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"GT: {len(gt_list)} | Pred: {len(pred_list)} | TP: {tp} FP: {fp} FN: {fn}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"P: {frame_precision:.4f} R: {frame_recall:.4f} F1: {frame_f1:.4f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

        if SAVE_IMAGE:
            save_name = f"frame_{frame_idx:04d}_compare.jpg"
            save_path = os.path.join(VIS_DIR, save_name)
            cv2.imwrite(save_path, vis)

        if SHOW_WINDOW:
            display = cv2.resize(vis, None, fx=0.5, fy=0.5)
            cv2.imshow("GT vs Prediction", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        if PRINT_PER_FRAME_LOG:
            print(
                f"[frame {frame_idx:04d}] "
                f"GT={len(gt_list)} Pred={len(pred_list)} "
                f"TP={tp} FP={fp} FN={fn} "
                f"P={frame_precision:.4f} R={frame_recall:.4f} F1={frame_f1:.4f}"
            )

    cap.release()
    cv2.destroyAllWindows()

    precision, recall, f1 = compute_prf1(total_tp, total_fp, total_fn)
    overall_class_match_ratio = safe_div(total_class_correct_pred, total_pred_eval)

    map50_per_class = {}
    map50_value = np.nan
    map50_95_per_class = {}
    map50_95_value = np.nan

    if COMPUTE_MAP50:
        map50_per_class, map50_value = compute_map_results(
            gt_buffer=ap_gt_buffer,
            pred_buffer=ap_pred_buffer,
            class_ids=class_ids,
            iou_thresholds=MAP50_IOU_THRESHOLDS
        )

    if COMPUTE_MAP50_95:
        map50_95_per_class, map50_95_value = compute_map_results(
            gt_buffer=ap_gt_buffer,
            pred_buffer=ap_pred_buffer,
            class_ids=class_ids,
            iou_thresholds=MAP50_95_IOU_THRESHOLDS
        )

    print("\n========== Detection Evaluation ==========")
    print(MODEL_PATH)
    print(f"USE_SAHI                    : {USE_SAHI}")
    print(f"USE_CENTER_ROI              : {USE_CENTER_ROI}")
    print(f"CROP_GT_TO_ROI              : {CROP_GT_TO_ROI}")
    print(f"USE_STABILIZATION           : {USE_STABILIZATION}")
    print(f"USE_CLASS_SMOOTHING         : {USE_CLASS_SMOOTHING}")
    print(f"USE_PERSISTENCE             : {USE_PERSISTENCE}")
    print(f"IOU_THRESH                  : {IOU_THRESH}")
    print(f"REQUIRE_CLASS_MATCH         : {REQUIRE_CLASS_MATCH}")
    print(f"SLICE_HEIGHT                : {SLICE_HEIGHT}")
    print(f"SLICE_WIDTH                 : {SLICE_WIDTH}")
    print(f"OVERLAP_HEIGHT_RATIO        : {OVERLAP_HEIGHT_RATIO}")
    print(f"OVERLAP_WIDTH_RATIO         : {OVERLAP_WIDTH_RATIO}")
    print(f"POSTPROCESS_TYPE            : {POSTPROCESS_TYPE}")
    print(f"POSTPROCESS_MATCH_METRIC    : {POSTPROCESS_MATCH_METRIC}")
    print(f"POSTPROCESS_MATCH_THRESHOLD : {POSTPROCESS_MATCH_THRESHOLD}")
    print(f"TP                          : {total_tp}")
    print(f"FP                          : {total_fp}")
    print(f"FN                          : {total_fn}")
    print(f"Precision                   : {precision:.6f}")
    print(f"Recall                      : {recall:.6f}")
    print(f"F1-score                    : {f1:.6f}")

    print("\n========== Prediction Class Correctness ==========")
    print(f"Predicted count (ripe+unripe) : {total_pred_eval}")
    print(f"Class-correct count           : {total_class_correct_pred}")
    print(f"Class-correct ratio           : {overall_class_match_ratio:.6f}")

    print("\n========== Per-class Metrics ==========")
    for cls_id in class_ids:
        cls_name = get_class_name(model_names, cls_id)
        cls_tp = per_class_stats[cls_id]["tp"]
        cls_fp = per_class_stats[cls_id]["fp"]
        cls_fn = per_class_stats[cls_id]["fn"]
        cls_p, cls_r, cls_f1 = compute_prf1(cls_tp, cls_fp, cls_fn)

        print(f"[{cls_name}]")
        print(f"TP                          : {cls_tp}")
        print(f"FP                          : {cls_fp}")
        print(f"FN                          : {cls_fn}")
        print(f"Precision                   : {cls_p:.6f}")
        print(f"Recall                      : {cls_r:.6f}")
        print(f"F1-score                    : {cls_f1:.6f}")

    print("\n========== AP / mAP ==========")
    if COMPUTE_MAP50:
        print(f"mAP50                       : {format_float(map50_value)}")
        for cls_id in class_ids:
            cls_name = get_class_name(model_names, cls_id)
            ap50 = map50_per_class[cls_id]["mean_ap"]
            print(f"AP50 [{cls_name}]                : {format_float(ap50)}")

    if COMPUTE_MAP50_95:
        print(f"mAP50-95                    : {format_float(map50_95_value)}")
        for cls_id in class_ids:
            cls_name = get_class_name(model_names, cls_id)
            ap5095 = map50_95_per_class[cls_id]["mean_ap"]
            print(f"AP50-95 [{cls_name}]             : {format_float(ap5095)}")

    if SAVE_SUMMARY_TXT:
        with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("Detection Evaluation Summary\n")
            f.write(f"USE_SAHI: {USE_SAHI}\n")
            f.write(f"USE_CENTER_ROI: {USE_CENTER_ROI}\n")
            f.write(f"CROP_GT_TO_ROI: {CROP_GT_TO_ROI}\n")
            f.write(f"USE_STABILIZATION: {USE_STABILIZATION}\n")
            f.write(f"USE_CLASS_SMOOTHING: {USE_CLASS_SMOOTHING}\n")
            f.write(f"USE_PERSISTENCE: {USE_PERSISTENCE}\n")
            f.write(f"IOU_THRESH: {IOU_THRESH}\n")
            f.write(f"REQUIRE_CLASS_MATCH: {REQUIRE_CLASS_MATCH}\n")
            f.write(f"SLICE_HEIGHT: {SLICE_HEIGHT}\n")
            f.write(f"SLICE_WIDTH: {SLICE_WIDTH}\n")
            f.write(f"OVERLAP_HEIGHT_RATIO: {OVERLAP_HEIGHT_RATIO}\n")
            f.write(f"OVERLAP_WIDTH_RATIO: {OVERLAP_WIDTH_RATIO}\n")
            f.write(f"POSTPROCESS_TYPE: {POSTPROCESS_TYPE}\n")
            f.write(f"POSTPROCESS_MATCH_METRIC: {POSTPROCESS_MATCH_METRIC}\n")
            f.write(f"POSTPROCESS_MATCH_THRESHOLD: {POSTPROCESS_MATCH_THRESHOLD}\n")
            f.write(f"TP: {total_tp}\n")
            f.write(f"FP: {total_fp}\n")
            f.write(f"FN: {total_fn}\n")
            f.write(f"Precision: {precision:.6f}\n")
            f.write(f"Recall: {recall:.6f}\n")
            f.write(f"F1-score: {f1:.6f}\n")
            f.write(f"Predicted count (ripe+unripe): {total_pred_eval}\n")
            f.write(f"Class-correct count: {total_class_correct_pred}\n")
            f.write(f"Class-correct ratio: {overall_class_match_ratio:.6f}\n")

            f.write("\nPer-class Metrics\n")
            for cls_id in class_ids:
                cls_name = get_class_name(model_names, cls_id)
                cls_tp = per_class_stats[cls_id]["tp"]
                cls_fp = per_class_stats[cls_id]["fp"]
                cls_fn = per_class_stats[cls_id]["fn"]
                cls_p, cls_r, cls_f1 = compute_prf1(cls_tp, cls_fp, cls_fn)

                f.write(f"[{cls_name}]\n")
                f.write(f"TP: {cls_tp}\n")
                f.write(f"FP: {cls_fp}\n")
                f.write(f"FN: {cls_fn}\n")
                f.write(f"Precision: {cls_p:.6f}\n")
                f.write(f"Recall: {cls_r:.6f}\n")
                f.write(f"F1-score: {cls_f1:.6f}\n")

            f.write("\nAP / mAP\n")
            if COMPUTE_MAP50:
                f.write(f"mAP50: {format_float(map50_value)}\n")
                for cls_id in class_ids:
                    cls_name = get_class_name(model_names, cls_id)
                    ap50 = map50_per_class[cls_id]["mean_ap"]
                    f.write(f"AP50 [{cls_name}]: {format_float(ap50)}\n")

            if COMPUTE_MAP50_95:
                f.write(f"mAP50-95: {format_float(map50_95_value)}\n")
                for cls_id in class_ids:
                    cls_name = get_class_name(model_names, cls_id)
                    ap5095 = map50_95_per_class[cls_id]["mean_ap"]
                    f.write(f"AP50-95 [{cls_name}]: {format_float(ap5095)}\n")

        print(f"\nSaved: {SUMMARY_TXT_PATH}")

    if SAVE_IMAGE:
        print(f"Visualization folder: {VIS_DIR}")

if __name__ == "__main__":
    main()
