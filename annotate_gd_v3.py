"""
GD 自动标注 v3 —— 每类独立 prompt + Healthy Wheat 负样本抑制 + 叶片掩码
5 类: Brown Rust(0), Black Rust(1), Septoria(2), Healthy Wheat(3), Powdery Mildew(4)
box_threshold=0.25, text_threshold=0.20
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
OUTPUT_DIR = PROJECT / "wheat_yolo_train_f3"
CHECKPOINT = PROJECT / "gd_v3_checkpoint_new.json"

# ============================================
# 5 类映射
# ============================================
CLASS_MAP = {
    "Brown Rust": 0,
    "Black Rust": 1,
    "Septoria": 2,
    "Healthy Wheat": 3,
    "Mildew": 4,
}
CLASS_NAMES = [None] * 5
for k, v in CLASS_MAP.items():
    CLASS_NAMES[v] = k

# ============================================
# 每类独立 prompt（每行一个独立 query）
# ============================================
CLASS_PROMPTS = {
    "Brown Rust": [
        "wheat leaf scattered with small oval brown rust pustules",
        "discrete tiny brown lesion spots on leaf surface",
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
}

# ============================================
# 推理参数
# ============================================
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
TRAIN_RATIO = 0.85
MIN_BOX_AREA = 100
MAX_BOX_AREA_RATIO = 0.6
MIN_LEAF_OVERLAP = 0.3
HEALTHY_IOU_SUPPRESS = 0.3
NMS_IOU = 0.5

# ============================================
# 加载 GD 模型
# ============================================
print(f"[{datetime.now():%H:%M:%S}] Loading Grounding DINO...")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-tiny"

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id, local_files_only=True
).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    model_id, local_files_only=True
)

vram = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
print(f"[{datetime.now():%H:%M:%S}] Model ready. Device={DEVICE}, VRAM={vram:.1f}GB")


# ============================================
# 叶片掩码 (HSV 绿色提取)
# ============================================
def leaf_mask_bgr(bgr_img):
    """返回叶片区域二值 mask，基于 HSV 绿色范围"""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 30, 30])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # 形态学闭运算填补小孔
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def box_overlap_ratio(box, mask):
    """计算框与叶片掩码的重叠比例"""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = mask.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2]
    return np.count_nonzero(crop) / crop.size if crop.size > 0 else 0.0


# ============================================
# GD 推理（多 prompt 合并）
# ============================================
def detect_with_prompts(img_bgr, prompts, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD):
    """对一张图用多个 prompt 推理，合并去重结果返回 [x1,y1,x2,y2]"""
    # 转 PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    all_boxes = []

    for prompt in prompts:
        try:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        except Exception:
            continue

        with torch.no_grad():
            try:
                outputs = model(**inputs)
            except Exception:
                continue

        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
        except Exception:
            continue

        for det in results[0]["boxes"]:
            all_boxes.append(det.tolist())

    return all_boxes


# ============================================
# 后处理
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


def filter_boxes(boxes, img_w, img_h, leaf_mask=None):
    """面积+宽高比+边缘+叶片重叠过滤 + NMS"""
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
            if ar > 5:
                continue
        if x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2:
            continue
        if leaf_mask is not None:
            overlap = box_overlap_ratio([x1, y1, x2, y2], leaf_mask)
            if overlap < MIN_LEAF_OVERLAP:
                continue
        filtered.append(box)

    return nms(filtered)


def suppress_healthy_boxes(disease_boxes, healthy_boxes, iou_threshold=HEALTHY_IOU_SUPPRESS):
    """移除与健康区域重叠过多的病变框"""
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
        print(f"[{datetime.now():%H:%M:%S}] Resume from checkpoint: {len(processed)} done")

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

    for idx, (img_path, cls_id, cls_name) in enumerate(remaining):
        # 读取图片
        try:
            with open(img_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_h, img_w = img_bgr.shape[:2]
        except Exception:
            continue

        # 叶片掩码
        leaf_mask_arr = leaf_mask_bgr(img_bgr)

        if cls_name == "Healthy Wheat":
            healthy_boxes_raw = detect_with_prompts(img_bgr, CLASS_PROMPTS["Healthy Wheat"])
            healthy_boxes = filter_boxes(healthy_boxes_raw, img_w, img_h, leaf_mask_arr)
            yolo_lines = []
        else:
            # 1. 病变检测
            disease_boxes_raw = detect_with_prompts(img_bgr, CLASS_PROMPTS[cls_name])
            # 2. 健康区域检测 (用于抑制)
            healthy_boxes_raw = detect_with_prompts(img_bgr, CLASS_PROMPTS["Healthy Wheat"])
            healthy_boxes = filter_boxes(healthy_boxes_raw, img_w, img_h, leaf_mask_arr)
            # 3. 过滤
            disease_filtered = filter_boxes(disease_boxes_raw, img_w, img_h, leaf_mask_arr)
            before_suppress = len(disease_filtered)
            # 4. 健康抑制
            disease_final = suppress_healthy_boxes(disease_filtered, healthy_boxes)
            healthy_suppressed += before_suppress - len(disease_final)
            box_total += len(disease_final)
            yolo_lines = boxes_to_yolo(disease_final, img_w, img_h, cls_id)

        # 写 label
        is_train = random.random() < TRAIN_RATIO
        split = "train" if is_train else "val"
        label_name = f"{cls_name}_{img_path.stem}.txt"
        label_path = OUTPUT_DIR / split / "labels" / label_name
        with open(label_path, "w", encoding="utf-8") as f:
            for line in yolo_lines:
                f.write(line + "\n")

        # 复制图片
        img_dst = OUTPUT_DIR / split / "images" / f"{cls_name}_{img_path.stem}{img_path.suffix}"
        if not img_dst.exists():
            shutil.copy2(img_path, img_dst)

        ok_count += 1
        if not yolo_lines and cls_name != "Healthy Wheat":
            empty_count += 1

        # 进度
        if (idx + 1) % 30 == 0 or idx < 3:
            elapsed = time.time() - t0
            spd = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - idx - 1) / spd if spd > 0 else 0
            nz = ok_count - empty_count
            avg = box_total / max(nz, 1)
            print(f"[{datetime.now():%H:%M:%S}] [{idx+1}/{len(remaining)}] "
                  f"avg={avg:.1f} box/img empty={empty_count} "
                  f"supp={healthy_suppressed} spd={spd:.1f}/s ETA={eta/60:.0f}min")

        # 每 200 张存 checkpoint
        if (idx + 1) % 200 == 0:
            processed.update(img.stem for img, _, _ in remaining[:idx+1])
            with open(CHECKPOINT, "w") as f:
                json.dump({"processed": list(processed), "last_idx": idx + 1,
                           "boxes_total": box_total, "healthy_suppressed": healthy_suppressed}, f)

    # 最终保存
    processed.update(img.stem for img, _, _ in remaining)
    with open(CHECKPOINT, "w") as f:
        json.dump({"processed": list(processed), "done": True,
                   "boxes_total": box_total, "healthy_suppressed": healthy_suppressed}, f)

    # 写 data.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images
nc: 5
names:
  0: Brown Rust
  1: Black Rust
  2: Septoria
  3: Healthy Wheat
  4: Powdery Mildew
"""
    with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    elapsed = time.time() - t0
    nz = ok_count - empty_count
    print(f"\n[{datetime.now():%H:%M:%S}] Done! {ok_count} images, "
          f"{box_total} boxes, avg={box_total/max(nz,1):.1f} box/img, "
          f"suppressed={healthy_suppressed}, {elapsed/60:.0f}min")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
