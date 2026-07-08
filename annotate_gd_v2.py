"""
Grounding DINO 全量标注 V2 — 简化策略
- 图片来自分类文件夹, 类别已知, GD 只负责找病灶位置
- 单一 prompt: "disease spots on wheat leaf"
- 低阈值高召回 + SAM 后续精修
- 断点续传
"""

import os
import json
import time
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# ========== 配置 ==========
ARCHIVE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\archive\Wheat_Disease")
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_detection_gd_v2")

SELECTED_CLASSES = [
    "Brown Rust",
    "Yellow Rust",
    "Black Rust",
    "Septoria",
    "Mildew",
    "Fusarium Head Blight",
    "Leaf Blight",
    "Healthy Wheat",
]

# 类名 → ID
CLASS_TO_ID = {name: i for i, name in enumerate(SELECTED_CLASSES)}

# ===== Grounding DINO 参数 =====
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BOX_THRESHOLD = 0.18       # 更低阈值保召回
TEXT_THRESHOLD = 0.15

# 统一 prompt — 只找病灶, 类别由文件夹决定
DISEASE_PROMPT = "disease spots . lesions . rust . blotch . mildew patches on wheat leaf"

# 进度文件
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"


def load_model():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"Loading {MODEL_ID}...")
    start = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to("cuda")
    model.eval()

    elapsed = time.time() - start
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"Model loaded in {elapsed:.1f}s. VRAM: {vram:.1f} GB")
    return model, processor


def detect_disease(model, processor, image_path):
    """单一推理 — 检出所有病灶区域"""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    inputs = processor(images=img, text=DISEASE_PROMPT, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(orig_h, orig_w)]
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    return boxes, scores, orig_w, orig_h


def boxes_to_yolo(boxes, img_w, img_h):
    """[x1,y1,x2,y2] → YOLO [cx,cy,w,h] 归一化"""
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.005, min(1.0, w))
        h = max(0.005, min(1.0, h))
        yolo_boxes.append([cx, cy, w, h])
    return yolo_boxes


def compute_box_area(box):
    return box[2] * box[3]


def nms_boxes(boxes_with_data, iou_threshold=0.5):
    """NMS 去重, 保留高置信度"""
    if len(boxes_with_data) <= 1:
        return boxes_with_data

    kept = []
    items = sorted(boxes_with_data, key=lambda x: x[2], reverse=True)

    while items:
        best = items.pop(0)
        kept.append(best)

        bx1 = best[0][0] - best[0][2] / 2
        by1 = best[0][1] - best[0][3] / 2
        bx2 = best[0][0] + best[0][2] / 2
        by2 = best[0][1] + best[0][3] / 2
        b_area = best[0][2] * best[0][3]

        filtered = []
        for item in items:
            cx, cy, w, h = item[0]
            ix1 = cx - w / 2
            iy1 = cy - h / 2
            ix2 = cx + w / 2
            iy2 = cy + h / 2

            inter_x1 = max(bx1, ix1)
            inter_y1 = max(by1, iy1)
            inter_x2 = min(bx2, ix2)
            inter_y2 = min(by2, iy2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                iou = inter_area / (b_area + w * h - inter_area + 1e-8)
                if iou < iou_threshold:
                    filtered.append(item)
            else:
                filtered.append(item)

        items = filtered

    return kept


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "total_boxes": 0, "total_images": 0}


def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def collect_all_images():
    """收集所有待标注图片, 按类别分组"""
    all_images = []
    for class_name in SELECTED_CLASSES:
        train_dir = ARCHIVE_DIR / "train" / class_name
        if not train_dir.exists():
            print(f"  [SKIP] {class_name}: directory not found")
            continue
        for img_path in train_dir.glob("*.*"):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                all_images.append((img_path, class_name))
    return all_images


def main():
    print("=" * 60)
    print("Grounding DINO V2 标注 — 类别来自文件夹, GD 只找病灶")
    print(f"类别: {len(SELECTED_CLASSES)} 类")
    print(f"模型: {MODEL_ID}")
    print(f"阈值: box={BOX_THRESHOLD}, text={TEXT_THRESHOLD}")
    print(f"Prompt: {DISEASE_PROMPT}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    # 写入 classes.txt
    with open(OUTPUT_DIR / "classes.txt", 'w', encoding='utf-8') as f:
        for name in SELECTED_CLASSES:
            f.write(f"{name}\n")

    all_images = collect_all_images()
    print(f"\n待处理图片: {len(all_images)} 张")

    # 按类别统计
    from collections import Counter
    class_counts = Counter(c for _, c in all_images)
    for cls, cnt in class_counts.most_common():
        print(f"  {cls}: {cnt}")

    # 加载进度
    ckpt = load_checkpoint()
    processed_set = set(ckpt["processed"])
    total_boxes = ckpt["total_boxes"]
    total_with_boxes = ckpt["total_images"]

    remaining = [(p, c) for p, c in all_images if p.name not in processed_set]

    if not remaining:
        print("全部已处理完成!")
        return

    print(f"\n已处理: {len(processed_set)}, 剩余: {len(remaining)}")
    if processed_set:
        print(f"已产出: {total_with_boxes} 张有框, {total_boxes} 个框, "
              f"平均 {total_boxes/max(total_with_boxes,1):.1f} 框/张")

    model, processor = load_model()

    start_time = time.time()
    success_count = 0
    no_det_count = 0
    error_count = 0

    for idx, (img_path, folder_class) in enumerate(remaining):
        try:
            img_name = img_path.name
            class_id = CLASS_TO_ID[folder_class]

            boxes, scores, orig_w, orig_h = detect_disease(model, processor, img_path)

            valid_detections = []

            if len(boxes) > 0:
                yolo_boxes = boxes_to_yolo(boxes, orig_w, orig_h)

                for yolo_box, score in zip(yolo_boxes, scores):
                    area = compute_box_area(yolo_box)

                    # 过滤极小框 (噪声) 和巨大框 (>90%, 几乎全图)
                    if area < 0.003:
                        continue
                    if area > 0.90:
                        continue

                    valid_detections.append((yolo_box, float(score)))

            # NMS 去重
            if valid_detections:
                valid_detections = nms_boxes(valid_detections, iou_threshold=0.5)

            # 保存
            if valid_detections:
                out_img_name = f"{folder_class}_{img_path.stem}.jpg"
                out_label_name = f"{folder_class}_{img_path.stem}.txt"

                out_img_path = OUTPUT_DIR / "images" / out_img_name
                if not out_img_path.exists():
                    img = Image.open(img_path)
                    if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                        img = img.convert('RGB')
                    img.save(out_img_path, quality=95)

                out_label_path = OUTPUT_DIR / "labels" / out_label_name
                with open(out_label_path, 'w') as f:
                    for yolo_box, score in valid_detections:
                        f.write(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} "
                                f"{yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

                total_boxes += len(valid_detections)
                total_with_boxes += 1
                success_count += 1
            else:
                no_det_count += 1

            processed_set.add(img_name)
            ckpt = {
                "processed": list(processed_set),
                "total_boxes": total_boxes,
                "total_images": total_with_boxes,
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            if (idx + 1) % 50 == 0:
                save_checkpoint(ckpt)
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - idx - 1) / speed if speed > 0 else 0
                avg_box = total_boxes / max(total_with_boxes, 1)
                print(f"  [{idx+1}/{len(remaining)}] "
                      f"ok={success_count} no_det={no_det_count} err={error_count} "
                      f"avg_box={avg_box:.1f} speed={speed:.2f}img/s ETA={eta/60:.0f}min")

        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"  [ERROR] {img_path.name}: {e}")
            processed_set.add(img_name)

    save_checkpoint(ckpt)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"标注完成!")
    print(f"  成功: {success_count} 张 | 未检出: {no_det_count} 张 | 出错: {error_count} 张")
    print(f"  总框数: {total_boxes} | 平均: {total_boxes/max(total_with_boxes,1):.1f} 框/张")
    print(f"  耗时: {elapsed/60:.1f} min | 输出: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
