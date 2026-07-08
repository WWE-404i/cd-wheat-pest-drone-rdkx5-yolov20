"""
Grounding DINO 全量标注 — 11831张 wheat disease 图片
输出真实 bbox, 替代中心缩框伪标注

优化:
  - 多类一次推理 (8类同时检测, 速度快8x, 类间竞争减少误检)
  - 断点续传 (checkpoint.json)
  - 低阈值保召回 (box=0.25, text=0.20)
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
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_detection_gd_full")

# 8类 (与当前训练一致)
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

# ===== Grounding DINO 参数 =====
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BOX_THRESHOLD = 0.25      # 低阈值保召回
TEXT_THRESHOLD = 0.20     # 文字匹配阈值

# ===== 多类联合 Prompt (一次推理检出所有病害) =====
# "." 分隔的短语会被 GD 独立匹配
COMBINED_PROMPT = (
    "brown rust spots on wheat leaf . "
    "yellow rust stripes on wheat leaf . "
    "black rust dark lesions on wheat . "
    "septoria leaf blotch brown spots yellow halo . "
    "powdery mildew white patches on wheat leaf . "
    "fusarium head blight bleached wheat spike . "
    "leaf blight dead spots on wheat leaf . "
    "healthy green wheat leaf no disease"
)

# 类别名到 prompt 短语的参考映射 (用于后处理匹配)
CLASS_PROMPT_MAP = {
    "Brown Rust": "brown rust spots on wheat leaf",
    "Yellow Rust": "yellow rust stripes on wheat leaf",
    "Black Rust": "black rust dark lesions on wheat",
    "Septoria": "septoria leaf blotch brown spots yellow halo",
    "Mildew": "powdery mildew white patches on wheat leaf",
    "Fusarium Head Blight": "fusarium head blight bleached wheat spike",
    "Leaf Blight": "leaf blight dead spots on wheat leaf",
    "Healthy Wheat": "healthy green wheat leaf no disease",
}

# 进度文件
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"


def load_model():
    """加载 Grounding DINO 模型"""
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


def detect_multi_class(model, processor, image_path):
    """一次推理检出所有病害类别"""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    inputs = processor(images=img, text=COMBINED_PROMPT, return_tensors="pt").to("cuda")

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
    labels = results["labels"]  # 返回的是文字标签列表

    return boxes, scores, labels, orig_w, orig_h


def match_label_to_class(label_str):
    """把 GD 返回的文字标签映射到我们的 8 类 ID"""
    label_lower = label_str.lower()

    # 关键词匹配
    keywords = {
        0: ["brown rust", "brown_rust"],
        1: ["yellow rust", "yellow_rust", "yellow rust stripes"],
        2: ["black rust", "black_rust", "dark lesion", "dark brown black"],
        3: ["septoria", "leaf blotch", "brown spots"],
        4: ["mildew", "powdery"],
        5: ["fusarium", "head blight", "bleached", "spike"],
        6: ["leaf blight", "dead spots"],
        7: ["healthy", "green wheat", "no disease"],
    }

    for class_id, kw_list in keywords.items():
        for kw in kw_list:
            if kw in label_lower:
                return class_id

    return -1  # 未匹配


def boxes_to_yolo(boxes, img_w, img_h):
    """[x1,y1,x2,y2] → YOLO [cx,cy,w,h] 归一化"""
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        # clamp
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.005, min(1.0, w))
        h = max(0.005, min(1.0, h))
        yolo_boxes.append([cx, cy, w, h])
    return yolo_boxes


def compute_box_area(box):
    """box = [cx, cy, w, h] 归一化, 返回面积占比"""
    return box[2] * box[3]


def nms_boxes(boxes_with_cls, iou_threshold=0.7):
    """对同一张图的检测框做 NMS 去重"""
    if len(boxes_with_cls) <= 1:
        return boxes_with_cls

    # 转为 [x1, y1, x2, y2] 格式做 IoU
    kept = []
    items = sorted(boxes_with_cls, key=lambda x: x[3], reverse=True)  # 按置信度排序

    while items:
        best = items.pop(0)
        kept.append(best)

        # 计算 best 与其余框的 IoU
        bx1 = best[1][0] - best[1][2] / 2
        by1 = best[1][1] - best[1][3] / 2
        bx2 = best[1][0] + best[1][2] / 2
        by2 = best[1][1] + best[1][3] / 2
        b_area = best[1][2] * best[1][3]

        filtered = []
        for item in items:
            cx, cy, w, h = item[1]
            ix1 = cx - w / 2
            iy1 = cy - h / 2
            ix2 = cx + w / 2
            iy2 = cy + h / 2

            # IoU
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
    """加载进度"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "total_boxes": 0, "total_images": 0}


def save_checkpoint(data):
    """保存进度"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def collect_all_images():
    """收集所有待标注图片"""
    all_images = []
    for class_name in SELECTED_CLASSES:
        train_dir = ARCHIVE_DIR / "train" / class_name
        if not train_dir.exists():
            print(f"  [SKIP] {class_name}: directory not found")
            continue
        images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.JPG")) + list(train_dir.glob("*.png"))
        for img_path in images:
            all_images.append((img_path, class_name))
    return all_images


def main():
    print("=" * 60)
    print("Grounding DINO 全量标注 — 小麦病害")
    print(f"类别: {len(SELECTED_CLASSES)} 类")
    print(f"模型: {MODEL_ID}")
    print(f"阈值: box={BOX_THRESHOLD}, text={TEXT_THRESHOLD}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    # 准备输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    # 写入 classes.txt
    with open(OUTPUT_DIR / "classes.txt", 'w', encoding='utf-8') as f:
        for name in SELECTED_CLASSES:
            f.write(f"{name}\n")

    # 收集所有图片
    all_images = collect_all_images()
    print(f"\n待处理图片: {len(all_images)} 张")

    # 加载进度
    ckpt = load_checkpoint()
    processed_set = set(ckpt["processed"])
    total_boxes = ckpt["total_boxes"]
    total_with_boxes = ckpt["total_images"]

    # 过滤已处理的
    remaining = [(p, c) for p, c in all_images if p.name not in processed_set]

    if not remaining:
        print("全部已处理完成!")
        return

    print(f"已处理: {len(processed_set)}, 剩余: {len(remaining)}")
    if processed_set:
        print(f"已产出: {total_with_boxes} 张有框, {total_boxes} 个框")

    # 加载模型
    model, processor = load_model()

    # ===== 逐张处理 =====
    start_time = time.time()
    success_count = 0
    error_count = 0
    no_det_count = 0

    for idx, (img_path, folder_class) in enumerate(remaining):
        try:
            img_name = img_path.name
            boxes, scores, labels, orig_w, orig_h = detect_multi_class(model, processor, img_path)

            valid_detections = []

            if len(boxes) > 0:
                # 匹配每个检测框到类别
                for box, score, label_str in zip(boxes, scores, labels):
                    class_id = match_label_to_class(str(label_str))
                    if class_id < 0:
                        continue  # 无法匹配类别, 跳过

                    yolo_box = boxes_to_yolo([box], orig_w, orig_h)[0]
                    area = compute_box_area(yolo_box)

                    # 过滤极小框 (噪声) 和巨大框 (>95%, 可能是全图)
                    if area < 0.005:
                        continue
                    if area > 0.95:
                        continue

                    valid_detections.append((class_id, yolo_box, float(score)))

            # NMS 去重
            if valid_detections:
                valid_detections = nms_boxes(valid_detections, iou_threshold=0.7)

            # 保存
            if valid_detections:
                out_img_name = f"{folder_class}_{img_path.stem}.jpg"
                out_label_name = f"{folder_class}_{img_path.stem}.txt"

                # 复制图片 (如果还没复制)
                out_img_path = OUTPUT_DIR / "images" / out_img_name
                if not out_img_path.exists():
                    img = Image.open(img_path)
                    if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                        img = img.convert('RGB')
                    img.save(out_img_path, quality=95)

                # 写 label
                out_label_path = OUTPUT_DIR / "labels" / out_label_name
                with open(out_label_path, 'w') as f:
                    for class_id, box, score in valid_detections:
                        f.write(f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

                total_boxes += len(valid_detections)
                total_with_boxes += 1
                success_count += 1
            else:
                no_det_count += 1

            # 更新 checkpoint
            processed_set.add(img_name)
            ckpt = {
                "processed": list(processed_set),
                "total_boxes": total_boxes,
                "total_images": total_with_boxes,
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 每 50 张保存一次进度
            if (idx + 1) % 50 == 0:
                save_checkpoint(ckpt)
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - idx - 1) / speed if speed > 0 else 0
                print(f"  [{idx+1}/{len(remaining)}] "
                      f"ok={success_count} no_det={no_det_count} err={error_count} "
                      f"speed={speed:.1f}img/s ETA={eta/60:.0f}min "
                      f"boxes_tot={total_boxes}")

        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"  [ERROR] {img_path.name}: {e}")
            # 标记为已处理, 避免卡在同一张
            processed_set.add(img_name)

    # 最终保存
    save_checkpoint(ckpt)

    # 汇总
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"标注完成!")
    print(f"  成功标注: {success_count} 张 (有框)")
    print(f"  未检出:   {no_det_count} 张 (无框)")
    print(f"  出错:     {error_count} 张")
    print(f"  总框数:   {total_boxes}")
    print(f"  耗时:     {elapsed/60:.1f} min")
    print(f"  输出:     {OUTPUT_DIR}")
    print(f"{'='*60}")

    # 生成 data.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: images
val: images
nc: {len(SELECTED_CLASSES)}
names: {SELECTED_CLASSES}
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    print("data.yaml 已生成")


if __name__ == "__main__":
    main()
