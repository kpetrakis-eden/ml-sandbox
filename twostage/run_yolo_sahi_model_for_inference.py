########## Inference script for the SAHI-tiled YOLO model.
# Tiles each image using the same parameters as train_yolo_sahi_model.py, runs YOLO on each
# tile, converts detections back to original image coordinates, applies NMS to remove
# duplicates from overlapping tiles, and saves annotated images + a JSON with detections.

import os
import json
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms as torch_nms
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH     = "/home/idroutsas/Desktop/Work/Scripts/Generalized_detection_model/runs/train/generalized_blueberry_sahi-3/weights/best.pt"    # <-- path to best.pt from train_yolo_sahi_model.py
INPUT_PATH     = "/home/idroutsas/Desktop/Work/Scripts/Generalized_detection_model/Data/raw_data_peru/images"      # <-- path to an image file or a directory of images
OUTPUT_DIR     = "/home/idroutsas/Desktop/Work/Scripts/Generalized_detection_model/results/inference_results/Peru"      # <-- directory to save annotated images and detections JSON

TILE_SIZE         = 640  # must match train_yolo_sahi_model.py
OVERLAP_RATIO     = 0.2  # must match train_yolo_sahi_model.py
CONF_THRESHOLD    = 0.260  # F1-optimal confidence threshold
NMS_IOU_THRESHOLD = 0.5    # IoU threshold for merging duplicate detections across tiles
BOX_COLOR         = "red"
BOX_WIDTH         = 2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ── Tiling (must mirror train_yolo_sahi_model.py) ─────────────────────────────

def get_tile_positions(img_w, img_h, tile_size, overlap_ratio):
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0)]
    stride = max(1, int(tile_size * (1 - overlap_ratio)))
    xs = list(range(0, img_w - tile_size, stride))
    if not xs or xs[-1] + tile_size < img_w:
        xs.append(max(0, img_w - tile_size))
    ys = list(range(0, img_h - tile_size, stride))
    if not ys or ys[-1] + tile_size < img_h:
        ys.append(max(0, img_h - tile_size))
    return [(x, y) for y in ys for x in xs]

# ── Per-tile inference ────────────────────────────────────────────────────────

def infer_tiles(model, img, conf_threshold):
    """
    Runs YOLO on each tile and returns all detections in original image coordinates.
    Each detection is [x1, y1, x2, y2, conf].
    """
    img_w, img_h = img.size
    tile_w = min(TILE_SIZE, img_w)
    tile_h = min(TILE_SIZE, img_h)
    detections = []

    for tile_x, tile_y in get_tile_positions(img_w, img_h, TILE_SIZE, OVERLAP_RATIO):
        tile = img.crop((tile_x, tile_y, tile_x + tile_w, tile_y + tile_h))
        results = model(tile, conf=conf_threshold, verbose=False)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                # Shift from tile coordinates to original image coordinates
                detections.append([x1 + tile_x, y1 + tile_y, x2 + tile_x, y2 + tile_y, conf])

    return detections

# ── NMS across tiles ──────────────────────────────────────────────────────────

def apply_nms(detections, iou_threshold):
    """
    Removes duplicate detections that arise from overlapping tiles.
    Returns filtered list of [x1, y1, x2, y2, conf].
    """
    if not detections:
        return []

    boxes  = torch.tensor([[d[0], d[1], d[2], d[3]] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in detections], dtype=torch.float32)
    keep   = torch_nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep.tolist()]

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_detections(img, detections):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x1, y1, x2, y2, _ in detections:
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
    return out

# ── Image collection ──────────────────────────────────────────────────────────

def collect_images(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    return [
        os.path.join(input_path, f)
        for f in sorted(os.listdir(input_path))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model  = YOLO(MODEL_PATH)
    images = collect_images(INPUT_PATH)
    print(f"Running inference on {len(images)} image(s)...\n")

    all_results = {}

    for img_path in images:
        img  = Image.open(img_path).convert("RGB")
        name = os.path.splitext(os.path.basename(img_path))[0]

        detections = infer_tiles(model, img, CONF_THRESHOLD)
        detections = apply_nms(detections, NMS_IOU_THRESHOLD)

        annotated = draw_detections(img, detections)
        annotated.save(os.path.join(OUTPUT_DIR, name + "_annotated.jpg"), "JPEG", quality=95)

        all_results[os.path.basename(img_path)] = [
            {"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3], "conf": round(d[4], 4)}
            for d in detections
        ]
        print(f"{os.path.basename(img_path)}: {len(detections)} detections")

    json_path = os.path.join(OUTPUT_DIR, "detections.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAnnotated images saved to: {OUTPUT_DIR}")
    print(f"Detections JSON saved to:  {json_path}")


if __name__ == "__main__":
    run()
