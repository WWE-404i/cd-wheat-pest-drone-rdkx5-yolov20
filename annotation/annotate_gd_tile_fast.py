"""
GD Tiling 自动标注 v2 (优化版) —— batch tiles + merged prompts + FP16
比原版快 3-5x，大图优势更明显
7 类: Brown Rust(0), Yellow Rust(1), Black Rust(2), Septoria(3),
      Healthy Wheat(4), Powdery Mildew(5), Fusarium Head Blight(6)
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
import random
import cv2
import shutil
from PIL import Image

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
KEEP_DIR = PROJECT / "image_audit" / "keep_短边大于等于640"
REVIEW_DIR = PROJECT / "image_audit" / "manual_review_480到639"
OUTPUT_DIR = PROJECT / "wheat_yolo_train_f5"
CHECKPOINT = PROJECT / "gd_tile_fast_checkpoint.json"

# ============================================
# 7 类映射
# ============================================
CLASS_MAP = {
    "Brown Rust": 0,
    "Yellow Rust": 1,
    "Black Rust": 2,
    "Septoria": 3,
    "Healthy Wheat": 4,
    "Mildew": 5,
    "Fusarium Head Blight": 6,
}

CLASS_NAMES = [None] * 7
for k, v in CLASS_MAP.items():
    CLASS_NAMES[v] = k
CLASS_NAMES[5] = "Powdery Mildew"

CLASS_PROMPTS = {
    "Brown Rust": [
        "wheat leaf scattered with small oval brown rust pustules",
        "discrete tiny brown lesion spots on leaf surface",
    ],
    "Yellow Rust": [
        "wheat leaf with yellow orange stripe rust pustules in rows",
        "bright yellow powdery rust stripes along leaf veins",
    ],
    "Black Rust": [
        "wheat leaf with large dark reddish-black clustered rust pustules",
        "irregular big dark rust patches on wheat blade",
    ],
    "Septoria": [
        "wheat leaf with small dark brown angular necrotic speck lesions",
        "scattered dark spot blotch on wheat leaf surface",
    ],
    "Healthy Wheat": [
        "smooth intact green wheat leaf with no spots, no mold, no rust lesions",
        "clean uniform leaf surface without disease pustules",
    ],
    "Mildew": [
        "wheat leaf covered with fluffy white powdery fungus coating",
        "patchy soft white mold lesions spread on leaf surface",
    ],
    "Fusarium Head Blight": [
        "bleached white pinkish wheat spike head with fusarium blight",
        "discolored pale wheat ear head with fungal infection",
    ],
}

# ============================================
# 参数
# ============================================
TILE_SIZE = 640
TILE_STRIDE = 512
BATCH_SIZE = 8          # tile batch size (tune: 4-16, 更大=更快但更吃显存)
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.10
TRAIN_RATIO = 0.85

# 后处理
MIN_BOX_AREA = 64
MAX_TILE_COVERAGE = 0.7
MAX_BOX_AREA_RATIO = 0.5
MAX_ASPECT_RATIO = 5
MIN_LEAF_OVERLAP = 0.15
NMS_IOU = 0.5
ENABLE_HEALTHY_SUPPRESS = False
HEALTHY_IOU_SUPPRESS = 0.3

# ============================================
# 加载 GD 模型
# ============================================
print(f"[{datetime.now():%H:%M:%S}] Loading Grounding DINO-tiny...")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")
model_id = "IDEA-Research/grounding-dino-tiny"

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id, local_files_only=True
).to(DEVICE)
processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

vram = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
print(f"[{datetime.now():%H:%M:%S}] Ready. {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}, "
      f"VRAM={vram:.1f}GB, AMP={USE_AMP}, Batch={BATCH_SIZE}")


# ============================================
# 叶片掩码
# ============================================
def leaf_mask_bgr(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 30, 30])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ============================================
# 优化核心: Batch tiles + merged prompts + FP16
# ============================================
def detect_batch(pil_images, combined_prompt):
    """
    一次推理处理多个 tile
    pil_images: list of PIL Image (所有 tile 尺寸相同=TILE_SIZE)
    combined_prompt: "prompt1 . prompt2 . ..." (合并所有 prompt)
    返回 list of [x1,y1,x2,y2] per tile
    """
    n = len(pil_images)
    all_results = []

    for i in range(0, n, BATCH_SIZE):
        batch = pil_images[i:i + BATCH_SIZE]
        texts = [combined_prompt] * len(batch)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                inputs = processor(images=batch, text=texts, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs)

            target_sizes = [(img.size[1], img.size[0]) for img in batch]
            results = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=target_sizes,
            )

        for r in results:
            boxes = []
            for det in r["boxes"]:
                boxes.append(det.tolist())
            all_results.append(boxes)

    return all_results


def _filter_tile_level(boxes, tile_w, tile_h):
    kept = []
    for box in boxes:
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        area = bw * bh
        if area < MIN_BOX_AREA:
            continue
        if area > tile_w * tile_h * MAX_TILE_COVERAGE:
            continue
        if bw > 0 and bh > 0:
            ar = max(bw / bh, bh / bw)
            if ar > MAX_ASPECT_RATIO:
                continue
        kept.append(box)
    return kept


def tile_and_detect(img_bgr, prompts, leaf_mask_arr):
    """
    分块检测（优化版）
    1. 收集所有有效 tile → PIL images
    2. batch 推理一次
    3. 映射回全图坐标
    """
    img_h, img_w = img_bgr.shape[:2]
    combined_prompt = " . ".join(prompts)

    # 小图直接检测
    if img_w <= TILE_SIZE and img_h <= TILE_SIZE:
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                inputs = processor(images=image, text=combined_prompt, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"],
                    box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                    target_sizes=[image.size[::-1]],
                )
            boxes = []
            for det in results[0]["boxes"]:
                boxes.append(det.tolist())
        return _filter_tile_level(boxes, img_w, img_h)

    # 生成 tiles
    tiles = []
    y = 0
    while y < img_h:
        if y + TILE_SIZE > img_h:
            y = max(0, img_h - TILE_SIZE)
        x = 0
        while x < img_w:
            if x + TILE_SIZE > img_w:
                x = max(0, img_w - TILE_SIZE)
            x2, y2 = min(x + TILE_SIZE, img_w), min(y + TILE_SIZE, img_h)
            tile_mask = leaf_mask_arr[y:y2, x:x2]
            leaf_pct = np.count_nonzero(tile_mask) / tile_mask.size if tile_mask.size > 0 else 0
            if leaf_pct >= 0.05:
                tiles.append((x, y, x2, y2))
            if x2 >= img_w:
                break
            x += TILE_STRIDE
        if y2 >= img_h:
            break
        y += TILE_STRIDE

    if not tiles:
        return []

    # 准备 batch
    pil_images = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile_bgr = img_bgr[ty1:ty2, tx1:tx2]
        img_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(img_rgb))

    # Batch 推理
    batch_results = detect_batch(pil_images, combined_prompt)

    # 映射回全图
    all_boxes = []
    for tile_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile_w, tile_h = tx2 - tx1, ty2 - ty1
        tile_boxes = _filter_tile_level(batch_results[tile_idx], tile_w, tile_h)
        for box in tile_boxes:
            bx1, by1, bx2, by2 = box
            all_boxes.append([
                bx1 + tx1, by1 + ty1,
                bx2 + tx1, by2 + ty1
            ])

    return all_boxes


# ============================================
# NMS & 过滤（与原版相同）
# ============================================
def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def nms(boxes, iou_threshold=NMS_IOU):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        filtered = []
        for b in boxes:
            if box_iou(best, b) < iou_threshold:
                filtered.append(b)
        boxes = filtered
    return keep


def box_overlap_ratio(box, mask):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = mask.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2]
    return np.count_nonzero(crop) / crop.size if crop.size > 0 else 0.0


def filter_boxes_global(boxes, img_w, img_h, leaf_mask=None):
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if area < MIN_BOX_AREA:
            continue
        if area > img_w * img_h * MAX_BOX_AREA_RATIO:
            continue
        if bw > 0 and bh > 0:
            ar = max(bw / bh, bh / bw)
            if ar > MAX_ASPECT_RATIO:
                continue
        if (x1 <= 3 and bw > img_w * 0.9) or (y1 <= 3 and bh > img_h * 0.9):
            continue
        if leaf_mask is not None:
            overlap = box_overlap_ratio([x1, y1, x2, y2], leaf_mask)
            if overlap < MIN_LEAF_OVERLAP:
                continue
        filtered.append(box)
    return nms(filtered)


def suppress_healthy_boxes(disease_boxes, healthy_boxes, iou_threshold=HEALTHY_IOU_SUPPRESS):
    if not healthy_boxes:
        return disease_boxes
    kept = []
    for db in disease_boxes:
        suppressed = False
        for hb in healthy_boxes:
            if box_iou(db, hb) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(db)
    return kept


def boxes_to_yolo(boxes, img_w, img_h, cls_id):
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        nw = (x2 - x1) / img_w
        nh = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


# ============================================
# 收集图片
# ============================================
def collect_images():
    items = []
    for source_dir in [KEEP_DIR, REVIEW_DIR]:
        if not source_dir.exists():
            continue
        for cls_dir in source_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            if cls_name not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls_name]
            for f in cls_dir.iterdir():
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    items.append((f, cls_id, cls_name))
    return items


# ============================================
# 主流程
# ============================================
def main():
    processed = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        processed = set(ckpt.get("processed", []))
        print(f"[{datetime.now():%H:%M:%S}] Resume: {len(processed)} done")

    items = collect_images()
    print(f"[{datetime.now():%H:%M:%S}] Collected {len(items)} images")

    remaining = [(img, cid, cname) for img, cid, cname in items
                 if img.stem not in processed]
    print(f"[{datetime.now():%H:%M:%S}] To annotate: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    random.shuffle(remaining)

    t0 = time.time()
    ok_count = 0
    box_total = 0
    empty_count = 0
    healthy_suppressed = 0
    tile_total = 0

    for idx, (img_path, cls_id, cls_name) in enumerate(remaining):
        try:
            with open(img_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_h, img_w = img_bgr.shape[:2]
        except Exception:
            continue

        leaf_mask_arr = leaf_mask_bgr(img_bgr)

        if cls_name == "Healthy Wheat":
            yolo_lines = []
        else:
            disease_boxes_raw = tile_and_detect(
                img_bgr, CLASS_PROMPTS[cls_name], leaf_mask_arr)
            disease_final = filter_boxes_global(
                disease_boxes_raw, img_w, img_h, leaf_mask_arr)

            if ENABLE_HEALTHY_SUPPRESS:
                healthy_raw = tile_and_detect(
                    img_bgr, CLASS_PROMPTS["Healthy Wheat"], leaf_mask_arr)
                healthy_boxes = filter_boxes_global(
                    healthy_raw, img_w, img_h, leaf_mask_arr)
                before = len(disease_final)
                disease_final = suppress_healthy_boxes(disease_final, healthy_boxes)
                healthy_suppressed += before - len(disease_final)

            box_total += len(disease_final)
            yolo_lines = boxes_to_yolo(disease_final, img_w, img_h, cls_id)

        is_train = random.random() < TRAIN_RATIO
        split = "train" if is_train else "val"
        label_name = f"{cls_name}_{img_path.stem}.txt"
        label_path = OUTPUT_DIR / split / "labels" / label_name
        with open(label_path, "w", encoding="utf-8") as f:
            for line in yolo_lines:
                f.write(line + "\n")

        img_dst = OUTPUT_DIR / split / "images" / f"{cls_name}_{img_path.stem}{img_path.suffix}"
        if not img_dst.exists():
            shutil.copy2(img_path, img_dst)

        ok_count += 1
        if not yolo_lines and cls_name != "Healthy Wheat":
            empty_count += 1

        if (idx + 1) % 30 == 0 or idx < 3:
            elapsed = time.time() - t0
            spd = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - idx - 1) / spd if spd > 0 else 0
            nz = ok_count - empty_count
            avg = box_total / max(nz, 1)
            print(f"[{datetime.now():%H:%M:%S}] [{idx+1}/{len(remaining)}] "
                  f"avg={avg:.1f} box/img empty={empty_count} "
                  f"spd={spd:.1f}/s ETA={eta/60:.0f}min")

        if (idx + 1) % 100 == 0:
            processed.update(img.stem for img, _, _ in remaining[:idx+1])
            with open(CHECKPOINT, "w") as f:
                json.dump({"processed": list(processed), "last_idx": idx + 1,
                           "boxes_total": box_total, "ok_count": ok_count,
                           "empty_count": empty_count}, f)

    processed.update(img.stem for img, _, _ in remaining)
    with open(CHECKPOINT, "w") as f:
        json.dump({"processed": list(processed), "done": True,
                   "boxes_total": box_total, "ok_count": ok_count,
                   "empty_count": empty_count}, f)

    # data.yaml
    names_yaml = "\n".join([f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)])
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images
nc: 7
names:
{names_yaml}
"""
    with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    elapsed = time.time() - t0
    nz = ok_count - empty_count
    print(f"\n[{datetime.now():%H:%M:%S}] Done! {ok_count} imgs, "
          f"{box_total} boxes, avg={box_total/max(nz,1):.1f} box/img, "
          f"empty={empty_count}, {elapsed/60:.0f}min")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
