"""
SAM 精修标注 — 用中心框作为 prompt, SAM 自动找病灶边界
优势: SAM 擅长分割, 中心框提供位置提示, 比 GD 强 100 倍
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
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_detection_sam")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

SELECTED_CLASSES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Mildew", "Fusarium Head Blight", "Leaf Blight", "Healthy Wheat",
]
CLASS_TO_ID = {name: i for i, name in enumerate(SELECTED_CLASSES)}

# SAM 参数
CENTER_BOX_RATIO = 0.60       # 中心框占图幅比例
SAM_IMAGE_SIZE = 1024         # SAM 处理分辨率
MIN_MASK_AREA = 0.01          # 最小 mask 面积 (1%)
MAX_MASK_AREA = 0.92          # 最大 mask 面积 (92%, 过滤整片叶子)

# 并行处理
BATCH_SIZE = 1  # SAM 逐张处理


def load_sam_model():
    """加载 SAM 模型"""
    print("Loading SAM model...")
    start = time.time()

    # 使用 ultralytics 内置 SAM (mobile-sam 快)
    from ultralytics import SAM
    model = SAM("mobile_sam.pt")
    model.to("cuda")
    model.eval()

    elapsed = time.time() - start
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"SAM loaded in {elapsed:.1f}s. VRAM: {vram:.1f} GB")
    return model


def get_center_box(img_w, img_h):
    """生成中心 60% 框 [x1, y1, x2, y2] (像素坐标)"""
    box_w = img_w * CENTER_BOX_RATIO
    box_h = img_h * CENTER_BOX_RATIO
    x1 = (img_w - box_w) / 2
    y1 = (img_h - box_h) / 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    return [x1, y1, x2, y2]


def sam_predict(model, image_path, box_xyxy):
    """SAM 推理: box prompt → mask"""
    img = Image.open(image_path).convert("RGB")

    # Resize 到 SAM 尺寸
    orig_w, orig_h = img.size
    scale_x = SAM_IMAGE_SIZE / orig_w
    scale_y = SAM_IMAGE_SIZE / orig_h

    img_resized = img.resize((SAM_IMAGE_SIZE, SAM_IMAGE_SIZE), Image.BILINEAR)
    img_np = np.array(img_resized)

    # 缩放 box 到 SAM 坐标
    box_sam = [
        box_xyxy[0] * scale_x,
        box_xyxy[1] * scale_y,
        box_xyxy[2] * scale_x,
        box_xyxy[3] * scale_y,
    ]

    # Ultralytics SAM API
    results = model(img_np, bboxes=[box_sam], device="cuda")

    if results and len(results) > 0:
        masks = results[0].masks
        if masks is not None and len(masks.data) > 0:
            # 取第一个 (最大置信度) mask
            mask = masks.data[0].cpu().numpy()

            # Resize mask 回原始尺寸
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
            mask = np.array(mask_img) > 127

            return mask

    return None


def mask_to_yolo_boxes(mask, img_w, img_h):
    """从 mask 提取 YOLO 格式框"""
    if mask.sum() == 0:
        return []

    # 找 mask 的连通区域
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)

    boxes = []
    for i in range(1, num_features + 1):
        region = labeled == i
        if region.sum() < 50:  # 过滤极小区域 (噪声)
            continue

        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # 转 YOLO
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.005, min(1.0, w))
        h = max(0.005, min(1.0, h))

        area = w * h
        if area < MIN_MASK_AREA or area > MAX_MASK_AREA:
            continue

        boxes.append([cx, cy, w, h])

    return boxes


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
    all_images = []
    for class_name in SELECTED_CLASSES:
        train_dir = ARCHIVE_DIR / "train" / class_name
        if not train_dir.exists():
            print(f"  [SKIP] {class_name}: not found")
            continue
        for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
            for img_path in train_dir.glob(ext):
                all_images.append((img_path, class_name))
    return all_images


def main():
    print("=" * 60)
    print("SAM 精修标注 — 中心框 → SAM 分割 → 精确 bbox")
    print(f"类别: {len(SELECTED_CLASSES)} 类")
    print(f"中心框: {CENTER_BOX_RATIO*100:.0f}%")
    print(f"Mask 面积: {MIN_MASK_AREA*100:.0f}%-{MAX_MASK_AREA*100:.0f}%")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "classes.txt", 'w', encoding='utf-8') as f:
        for name in SELECTED_CLASSES:
            f.write(f"{name}\n")

    all_images = collect_all_images()
    print(f"\n待处理: {len(all_images)} 张")

    # 按类统计
    from collections import Counter
    for cls, cnt in Counter(c for _, c in all_images).most_common():
        print(f"  {cls}: {cnt}")

    ckpt = load_checkpoint()
    processed_set = set(ckpt["processed"])
    remaining = [(p, c) for p, c in all_images if p.name not in processed_set]

    if not remaining:
        print("全部完成!")
        return

    print(f"\n已处理: {len(processed_set)}, 剩余: {len(remaining)}")
    total_boxes = ckpt["total_boxes"]
    total_with_boxes = ckpt["total_images"]
    if total_with_boxes > 0:
        print(f"已产出: {total_with_boxes} 张有框, {total_boxes} 个框, "
              f"平均 {total_boxes/total_with_boxes:.1f} 框/张")

    model = load_sam_model()

    start_time = time.time()
    ok_count = 0
    no_mask = 0
    err_count = 0

    for idx, (img_path, folder_class) in enumerate(remaining):
        try:
            class_id = CLASS_TO_ID[folder_class]
            img = Image.open(img_path)
            orig_w, orig_h = img.size

            # 中心框 → SAM
            center_box = get_center_box(orig_w, orig_h)
            mask = sam_predict(model, img_path, center_box)

            if mask is None or mask.sum() == 0:
                no_mask += 1
                processed_set.add(img_path.name)
                continue

            # Mask → bboxes
            yolo_boxes = mask_to_yolo_boxes(mask, orig_w, orig_h)

            if yolo_boxes:
                out_img_name = f"{folder_class}_{img_path.stem}.jpg"
                out_label_name = f"{folder_class}_{img_path.stem}.txt"

                out_img_path = OUTPUT_DIR / "images" / out_img_name
                if not out_img_path.exists():
                    if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                        img = img.convert('RGB')
                    img.save(out_img_path, quality=95)

                out_label_path = OUTPUT_DIR / "labels" / out_label_name
                with open(out_label_path, 'w') as f:
                    for box in yolo_boxes:
                        f.write(f"{class_id} {box[0]:.6f} {box[1]:.6f} "
                                f"{box[2]:.6f} {box[3]:.6f}\n")

                total_boxes += len(yolo_boxes)
                total_with_boxes += 1
                ok_count += 1
            else:
                no_mask += 1

            processed_set.add(img_path.name)

            # 每 100 张存 checkpoint
            if (idx + 1) % 100 == 0:
                ckpt = {
                    "processed": list(processed_set),
                    "total_boxes": total_boxes,
                    "total_images": total_with_boxes,
                    "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_checkpoint(ckpt)
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - idx - 1) / speed if speed > 0 else 0
                avg = total_boxes / max(total_with_boxes, 1)
                print(f"  [{idx+1}/{len(remaining)}] "
                      f"ok={ok_count} no_mask={no_mask} err={err_count} "
                      f"avg={avg:.1f}box/img speed={speed:.2f}img/s ETA={eta/60:.0f}min")

        except Exception as e:
            err_count += 1
            if err_count <= 10:
                print(f"  [ERR] {img_path.name}: {e}")
            processed_set.add(img_path.name)
            if err_count > 50:
                print("错误太多, 终止!")
                break

    # 最终保存
    ckpt = {
        "processed": list(processed_set),
        "total_boxes": total_boxes,
        "total_images": total_with_boxes,
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_checkpoint(ckpt)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SAM 标注完成!")
    print(f"  成功: {ok_count} 张 ({ok_count/max(len(remaining),1)*100:.1f}%)")
    print(f"  无mask: {no_mask} 张")
    print(f"  出错: {err_count} 张")
    print(f"  总框: {total_boxes} | 平均: {total_boxes/max(total_with_boxes,1):.1f} 框/张")
    print(f"  耗时: {elapsed/60:.1f} min")
    print(f"{'='*60}")

    # 生成 data.yaml
    yaml = f"""path: {OUTPUT_DIR}
train: images
val: images
nc: {len(SELECTED_CLASSES)}
names: {SELECTED_CLASSES}
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml)
    print("data.yaml 已生成")


if __name__ == "__main__":
    main()
