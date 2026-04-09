import os
import re
import json
import csv
import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\trained_merge\yolo26n_640\weights\best.pt"
VIDEO_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\0407.mp4"
COCO_JSON_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric_coco\train\_annotations.coco.json"

OUTPUT_DIR = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\result4"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

PRED_CSV_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
MATCH_CSV_PATH = os.path.join(OUTPUT_DIR, "matches.csv")
FRAME_SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "frame_summary.csv")
SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "summary.txt")

IMGSZ = 768
CONF_THRES = 0.2
DEVICE = "0"

USE_ZED_ONE_SIDE = False
USE_LEFT_ONLY = False

MIN_BOX_AREA = 2000
MIN_BOX_W = 50
MIN_BOX_H = 50

IOU_THRESH = 0.3
REQUIRE_CLASS_MATCH = False

SHOW_WINDOW = False
SAVE_IMAGE = True

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

def preprocess_frame(frame):
    if USE_ZED_ONE_SIDE:
        h, w = frame.shape[:2]
        half_w = w // 2
        if USE_LEFT_ONLY:
            frame = frame[:, :half_w]
        else:
            frame = frame[:, half_w:]
    return frame

def filter_pred_boxes(xyxy, confs, clss):
    filtered = []

    for box, conf, cls_id in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = map(float, box)
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if area < MIN_BOX_AREA:
            continue
        if bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue

        filtered.append({
            "bbox": [x1, y1, x2, y2],
            "conf": float(conf),
            "cls_id": int(cls_id),
        })

    return filtered

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

def predict_on_frame(model, frame):
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

    return filter_pred_boxes(xyxy, confs, clss)

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
            "pred": pred_list[pi],
        })

    unmatched_gt = [gt_list[i] for i in range(len(gt_list)) if i not in matched_gt]
    unmatched_pred = [pred_list[i] for i in range(len(pred_list)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred

def draw_boxes(frame, gt_list, pred_list, model_names):
    vis = frame.copy()

    for gt in gt_list:
        x1, y1, x2, y2 = map(int, gt["bbox"])
        cls_name = gt["class_name"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(
            vis,
            f"GT: {cls_name}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    for pred in pred_list:
        x1, y1, x2, y2 = map(int, pred["bbox"])
        cls_id = pred["cls_id"]
        conf = pred["conf"]
        cls_name = model_names.get(cls_id, str(cls_id))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"Pred: {cls_name} {conf:.2f}",
            (x1, min(y2 + 25, vis.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

    return vis

def main():
    ensure_dir(OUTPUT_DIR)
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
    model = YOLO(MODEL_PATH)
    model_names = model.names

    print("YOLO class names:", model_names)
    print("COCO category map:", cat_id_to_name)

    gt_to_yolo_cls = build_gt_to_yolo_class_map(cat_id_to_name, model_names)
    print("GT category_id -> YOLO cls_id:", gt_to_yolo_cls)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    pred_rows = []
    match_rows = []
    frame_summary_rows = []

    for frame_idx in eval_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"[Warning] Failed to read frame {frame_idx}")
            continue

        frame = preprocess_frame(frame)

        gt_list = gt_by_frame[frame_idx]
        pred_list = predict_on_frame(model, frame)

        matches, unmatched_gt, unmatched_pred = greedy_match(
            gt_list,
            pred_list,
            gt_to_yolo_cls=gt_to_yolo_cls
        )

        tp = len(matches)
        fn = len(unmatched_gt)
        fp = len(unmatched_pred)

        total_tp += tp
        total_fn += fn
        total_fp += fp

        frame_precision = tp / (tp + fp + 1e-9)
        frame_recall = tp / (tp + fn + 1e-9)
        frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall + 1e-9)

        original_name = gt_list[0]["original_name"] if len(gt_list) > 0 else f"frame_{frame_idx:04d}.jpg"

        frame_summary_rows.append({
            "frame_idx": frame_idx,
            "file_name": original_name,
            "gt_count": len(gt_list),
            "pred_count": len(pred_list),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": frame_precision,
            "recall": frame_recall,
            "f1": frame_f1
        })

        for pred in pred_list:
            x1, y1, x2, y2 = pred["bbox"]
            pred_rows.append({
                "frame_idx": frame_idx,
                "file_name": original_name,
                "pred_class_id": pred["cls_id"],
                "pred_class_name": model_names.get(pred["cls_id"], str(pred["cls_id"])),
                "conf": pred["conf"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        for m in matches:
            gx1, gy1, gx2, gy2 = m["gt"]["bbox"]
            px1, py1, px2, py2 = m["pred"]["bbox"]

            match_rows.append({
                "frame_idx": frame_idx,
                "file_name": original_name,
                "status": "TP",
                "iou": m["iou"],
                "gt_ann_id": m["gt"]["ann_id"],
                "gt_class_name": m["gt"]["class_name"],
                "gt_x1": gx1, "gt_y1": gy1, "gt_x2": gx2, "gt_y2": gy2,
                "pred_class_id": m["pred"]["cls_id"],
                "pred_class_name": model_names.get(m["pred"]["cls_id"], str(m["pred"]["cls_id"])),
                "pred_conf": m["pred"]["conf"],
                "pred_x1": px1, "pred_y1": py1, "pred_x2": px2, "pred_y2": py2,
            })

        for gt in unmatched_gt:
            gx1, gy1, gx2, gy2 = gt["bbox"]
            match_rows.append({
                "frame_idx": frame_idx,
                "file_name": original_name,
                "status": "FN",
                "iou": "",
                "gt_ann_id": gt["ann_id"],
                "gt_class_name": gt["class_name"],
                "gt_x1": gx1, "gt_y1": gy1, "gt_x2": gx2, "gt_y2": gy2,
                "pred_class_id": "",
                "pred_class_name": "",
                "pred_conf": "",
                "pred_x1": "", "pred_y1": "", "pred_x2": "", "pred_y2": "",
            })

        for pred in unmatched_pred:
            px1, py1, px2, py2 = pred["bbox"]
            match_rows.append({
                "frame_idx": frame_idx,
                "file_name": original_name,
                "status": "FP",
                "iou": "",
                "gt_ann_id": "",
                "gt_class_name": "",
                "gt_x1": "", "gt_y1": "", "gt_x2": "", "gt_y2": "",
                "pred_class_id": pred["cls_id"],
                "pred_class_name": model_names.get(pred["cls_id"], str(pred["cls_id"])),
                "pred_conf": pred["conf"],
                "pred_x1": px1, "pred_y1": py1, "pred_x2": px2, "pred_y2": py2,
            })

        vis = draw_boxes(frame, gt_list, pred_list, model_names)

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

        print(
            f"[frame {frame_idx:04d}] "
            f"GT={len(gt_list)} Pred={len(pred_list)} "
            f"TP={tp} FP={fp} FN={fn}"
        )

    cap.release()
    cv2.destroyAllWindows()

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("\n========== Detection Evaluation ==========")
    print(f"IOU_THRESH          : {IOU_THRESH}")
    print(f"REQUIRE_CLASS_MATCH : {REQUIRE_CLASS_MATCH}")
    print(f"TP                  : {total_tp}")
    print(f"FP                  : {total_fp}")
    print(f"FN                  : {total_fn}")
    print(f"Precision           : {precision:.6f}")
    print(f"Recall              : {recall:.6f}")
    print(f"F1-score            : {f1:.6f}")

    with open(PRED_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx", "file_name",
                "pred_class_id", "pred_class_name", "conf",
                "x1", "y1", "x2", "y2"
            ]
        )
        writer.writeheader()
        writer.writerows(pred_rows)

    with open(MATCH_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx", "file_name", "status", "iou",
                "gt_ann_id", "gt_class_name", "gt_x1", "gt_y1", "gt_x2", "gt_y2",
                "pred_class_id", "pred_class_name", "pred_conf",
                "pred_x1", "pred_y1", "pred_x2", "pred_y2"
            ]
        )
        writer.writeheader()
        writer.writerows(match_rows)

    with open(FRAME_SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx", "file_name",
                "gt_count", "pred_count",
                "tp", "fp", "fn",
                "precision", "recall", "f1"
            ]
        )
        writer.writeheader()
        writer.writerows(frame_summary_rows)

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("Detection Evaluation Summary\n")
        f.write(f"IOU_THRESH: {IOU_THRESH}\n")
        f.write(f"REQUIRE_CLASS_MATCH: {REQUIRE_CLASS_MATCH}\n")
        f.write(f"TP: {total_tp}\n")
        f.write(f"FP: {total_fp}\n")
        f.write(f"FN: {total_fn}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1-score: {f1:.6f}\n")

    print("\nSaved:")
    print(f"- {PRED_CSV_PATH}")
    print(f"- {MATCH_CSV_PATH}")
    print(f"- {FRAME_SUMMARY_CSV_PATH}")
    print(f"- {SUMMARY_TXT_PATH}")
    print(f"- Visualization folder: {VIS_DIR}")

if __name__ == "__main__":
    main()
