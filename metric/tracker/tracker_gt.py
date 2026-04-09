import json
import os
import re
import math
import csv
from collections import defaultdict

import cv2

JSON_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\metric_coco\train\_annotations.coco.json"
IMAGE_DIR = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\metric_coco\train"

OUT_CSV = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\metric_coco\train\track\tracking_gt_auto.csv"
OUT_JSON = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\metric_coco\train\track\tracking_gt_with_trackid.json"
OUT_VIS_DIR = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\metric\metric_coco\train\track\gt_vis"

IOU_THRESH = 0.10
MAX_CENTER_DIST = 150
REQUIRE_CLASS_MATCH = True
MAX_MISSING_FRAMES = 10

DRAW_THICKNESS = 3
FONT_SCALE = 0.8
TEXT_THICKNESS = 2


def extract_frame_number(name: str):
    m = re.search(r'frame_(\d+)', name)
    return int(m.group(1)) if m else -1


def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    x1, y1, x2, y2 = bbox_xywh_to_xyxy(box1)
    x1b, y1b, x2b, y2b = bbox_xywh_to_xyxy(box2)

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def center_of_bbox(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def center_distance(box1, box2):
    c1 = center_of_bbox(box1)
    c2 = center_of_bbox(box2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def make_color_from_id(track_id: int):
    r = (37 * track_id) % 200 + 30
    g = (17 * track_id) % 200 + 30
    b = (97 * track_id) % 200 + 30
    return (b, g, r)


def draw_label_box(img, text, x, y, color):
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS
    )
    x1 = int(x)
    y1 = int(max(0, y - th - baseline - 6))
    x2 = int(x + tw + 8)
    y2 = int(y)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 4, y2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        TEXT_THICKNESS,
        cv2.LINE_AA
    )


os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(OUT_VIS_DIR, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = {c["id"]: c["name"] for c in coco["categories"]}

image_info = {}
for img in images:
    original_name = img.get("extra", {}).get("name", img["file_name"])
    image_info[img["id"]] = {
        "file_name_exported": img["file_name"],
        "original_name": original_name,
        "frame_num": extract_frame_number(original_name),
        "width": img["width"],
        "height": img["height"],
    }

ann_by_image = defaultdict(list)
for ann in annotations:
    raw_bbox = ann["bbox"]
    bbox = [float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])]

    ann_by_image[ann["image_id"]].append({
        "ann_id": ann["id"],
        "image_id": ann["image_id"],
        "category_id": ann["category_id"],
        "bbox": bbox,
        "area": float(ann.get("area", bbox[2] * bbox[3])),
        "iscrowd": ann.get("iscrowd", 0),
        "segmentation": ann.get("segmentation", None),
    })

sorted_image_ids = sorted(
    image_info.keys(),
    key=lambda iid: image_info[iid]["frame_num"]
)

next_track_id = 1
active_tracks = {}
results = []

for frame_idx, image_id in enumerate(sorted_image_ids):
    curr_anns = ann_by_image.get(image_id, [])
    curr_file = image_info[image_id]["original_name"]
    raw_frame_num = image_info[image_id]["frame_num"]

    current_objects = []
    for ann in curr_anns:
        current_objects.append({
            **ann,
            "track_id": None
        })

    valid_track_ids = []
    for track_id, track in active_tracks.items():
        gap = frame_idx - track["last_seen_idx"]
        if 1 <= gap <= MAX_MISSING_FRAMES + 1:
            valid_track_ids.append(track_id)

    used_tracks = set()
    used_curr = set()
    candidates = []

    for track_id in valid_track_ids:
        track = active_tracks[track_id]
        for ci, curr_obj in enumerate(current_objects):
            if REQUIRE_CLASS_MATCH and track["category_id"] != curr_obj["category_id"]:
                continue

            iou = compute_iou(track["bbox"], curr_obj["bbox"])
            dist = center_distance(track["bbox"], curr_obj["bbox"])

            if iou >= IOU_THRESH or dist <= MAX_CENTER_DIST:
                gap = frame_idx - track["last_seen_idx"]
                score = iou - 0.001 * dist - 0.05 * (gap - 1)
                candidates.append((score, track_id, ci))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for score, track_id, ci in candidates:
        if track_id in used_tracks or ci in used_curr:
            continue
        current_objects[ci]["track_id"] = track_id
        used_tracks.add(track_id)
        used_curr.add(ci)

    for obj in current_objects:
        if obj["track_id"] is None:
            obj["track_id"] = next_track_id
            next_track_id += 1

    for obj in current_objects:
        track_id = obj["track_id"]
        active_tracks[track_id] = {
            "track_id": track_id,
            "bbox": obj["bbox"],
            "category_id": obj["category_id"],
            "last_seen_idx": frame_idx,
        }

        results.append({
            "frame_idx": frame_idx,
            "frame_num": raw_frame_num,
            "file_name": curr_file,
            "image_id": obj["image_id"],
            "ann_id": obj["ann_id"],
            "track_id": obj["track_id"],
            "category_id": obj["category_id"],
            "category_name": categories.get(obj["category_id"], str(obj["category_id"])),
            "x": obj["bbox"][0],
            "y": obj["bbox"][1],
            "w": obj["bbox"][2],
            "h": obj["bbox"][3],
        })

    expired_ids = []
    for track_id, track in active_tracks.items():
        if frame_idx - track["last_seen_idx"] > MAX_MISSING_FRAMES + 1:
            expired_ids.append(track_id)

    for track_id in expired_ids:
        del active_tracks[track_id]

with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame_idx", "frame_num", "file_name", "image_id", "ann_id",
            "track_id", "category_id", "category_name",
            "x", "y", "w", "h"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"[1/3] CSV saved: {OUT_CSV}")

annid_to_trackid = {row["ann_id"]: row["track_id"] for row in results}

coco_with_track = json.loads(json.dumps(coco))
for ann in coco_with_track["annotations"]:
    ann["track_id"] = annid_to_trackid.get(ann["id"], -1)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco_with_track, f, ensure_ascii=False, indent=2)

print(f"[2/3] JSON saved: {OUT_JSON}")

rows_by_image = defaultdict(list)
for row in results:
    rows_by_image[row["image_id"]].append(row)

num_saved = 0
num_missing = 0

for image_id in sorted_image_ids:
    info = image_info[image_id]

    candidate_paths = [
        os.path.join(IMAGE_DIR, info["original_name"]),
        os.path.join(IMAGE_DIR, info["file_name_exported"]),
    ]

    img_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            img_path = p
            break

    if img_path is None:
        print(f"[WARN] Missing image file: image_id={image_id}, candidates={candidate_paths}")
        num_missing += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Failed to read image: {img_path}")
        num_missing += 1
        continue

    rows = rows_by_image.get(image_id, [])

    for row in rows:
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["w"])
        h = int(row["h"])
        x2 = x + w
        y2 = y + h

        track_id = int(row["track_id"])
        cls_name = row["category_name"]

        color = make_color_from_id(track_id)

        cv2.rectangle(img, (x, y), (x2, y2), color, DRAW_THICKNESS)

        label = f"ID {track_id} | GT {cls_name}"
        label_y = y - 5 if y - 5 > 20 else y + 25
        draw_label_box(img, label, x, label_y, color)

    out_name = info["original_name"]
    out_path = os.path.join(OUT_VIS_DIR, out_name)
    cv2.imwrite(out_path, img)
    num_saved += 1

print(f"[3/3] Visualization saved: {OUT_VIS_DIR}")
print(f"Saved images: {num_saved}")
print(f"Missing images: {num_missing}")
print(f"Total tracks: {next_track_id - 1}")
print(f"Total annotations: {len(results)}")
