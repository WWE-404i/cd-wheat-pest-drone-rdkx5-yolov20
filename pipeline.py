"""
一条龙: SAM 标注 → 切分训练集 → YOLOv26n 训练
断点续传, 可随时中断重跑
"""

import os
import json
import time
import shutil
import random
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# ========== 配置 ==========
ARCHIVE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\archive\Wheat_Disease")
WORK_DIR = Path(r"D:\python\wheat-yolo")  # junction 绕过 ultralytics 中文路径问题
SAM_OUTPUT = WORK_DIR / "wheat_detection_sam"
TRAIN_OUTPUT = WORK_DIR / "wheat_yolo_train"
CHECKPOINT_FILE = SAM_OUTPUT / "checkpoint.json"

SELECTED_CLASSES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Mildew", "Fusarium Head Blight", "Leaf Blight", "Healthy Wheat",
]
CLASS_TO_ID = {name: i for i, name in enumerate(SELECTED_CLASSES)}

# SAM 参数
CENTER_BOX_RATIO = 0.60
SAM_IMAGE_SIZE = 1024
MIN_MASK_AREA = 0.01
MAX_MASK_AREA = 0.92

# 训练参数
TRAIN_RATIO = 0.85
EPOCHS = 200
IMSZ = 640
BATCH = 40

# ==================== Stage 1: SAM 标注 ====================

def load_sam_model():
    print("Loading Mobile SAM...")
    from ultralytics import SAM
    model = SAM("mobile_sam.pt")
    model.to("cuda")
    model.eval()
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"SAM ready. VRAM: {vram:.1f} GB")
    return model


def get_center_box(img_w, img_h):
    box_w = img_w * CENTER_BOX_RATIO
    box_h = img_h * CENTER_BOX_RATIO
    x1, y1 = (img_w - box_w) / 2, (img_h - box_h) / 2
    return [x1, y1, x1 + box_w, y1 + box_h]


def sam_predict(model, image_path, box_xyxy):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale_x = SAM_IMAGE_SIZE / orig_w
    scale_y = SAM_IMAGE_SIZE / orig_h
    img_resized = img.resize((SAM_IMAGE_SIZE, SAM_IMAGE_SIZE), Image.BILINEAR)
    box_sam = [box_xyxy[0] * scale_x, box_xyxy[1] * scale_y,
               box_xyxy[2] * scale_x, box_xyxy[3] * scale_y]

    results = model(np.array(img_resized), bboxes=[box_sam], device="cuda")
    if results and len(results) > 0:
        masks = results[0].masks
        if masks is not None and len(masks.data) > 0:
            mask = masks.data[0].cpu().numpy()
            # 保持 SAM 尺寸做连通域分析, 避免大图 resize 后 ndimage.label 极慢
            return mask > 0.5, scale_x, scale_y, orig_w, orig_h
    return None, None, None, None, None


def mask_to_boxes(mask, scale_x, scale_y, orig_w, orig_h):
    if mask is None or mask.sum() == 0:
        return []
    from scipy import ndimage
    labeled, n = ndimage.label(mask)
    boxes = []
    for i in range(1, n + 1):
        region = labeled == i
        if region.sum() < 50:
            continue
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cx = max(0.0, min(1.0, ((x1 + x2) / 2) / img_w))
        cy = max(0.0, min(1.0, ((y1 + y2) / 2) / img_h))
        w = max(0.005, min(1.0, (x2 - x1) / img_w))
        h = max(0.005, min(1.0, (y2 - y1) / img_h))
        area = w * h
        if MIN_MASK_AREA <= area <= MAX_MASK_AREA:
            boxes.append([cx, cy, w, h])
    return boxes


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "total_boxes": 0, "total_images": 0}


def save_checkpoint(data):
    SAM_OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def collect_images():
    imgs = []
    for cls in SELECTED_CLASSES:
        d = ARCHIVE_DIR / "train" / cls
        if d.exists():
            for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
                for p in d.glob(ext):
                    imgs.append((p, cls))
    return imgs


def stage1_sam():
    print("\n" + "=" * 60)
    print("STAGE 1: SAM 标注")
    print("=" * 60)

    SAM_OUTPUT.mkdir(parents=True, exist_ok=True)
    # 按类别分子目录, 避免 NTFS 单目录文件数过多导致性能下降
    for cls_name in SELECTED_CLASSES:
        (SAM_OUTPUT / "images" / cls_name).mkdir(parents=True, exist_ok=True)
        (SAM_OUTPUT / "labels" / cls_name).mkdir(parents=True, exist_ok=True)

    with open(SAM_OUTPUT / "classes.txt", 'w', encoding='utf-8') as f:
        for n in SELECTED_CLASSES:
            f.write(f"{n}\n")

    all_imgs = collect_images()
    print(f"总图片: {len(all_imgs)}")

    ckpt = load_checkpoint()
    done = set(ckpt["processed"])
    remaining = [(p, c) for p, c in all_imgs if p.name not in done]

    if not remaining:
        print("SAM 阶段已完成!")
        return ckpt

    print(f"已处理: {len(done)}, 剩余: {len(remaining)}")
    total_boxes = ckpt["total_boxes"]
    total_with = ckpt["total_images"]

    model = load_sam_model()
    t0 = time.time()
    ok = no_mask = err = 0

    for idx, (img_path, cls_name) in enumerate(remaining):
        try:
            cid = CLASS_TO_ID[cls_name]
            img = Image.open(img_path)
            ow, oh = img.size

            cb = get_center_box(ow, oh)
            mask, sx, sy, sw, sh = sam_predict(model, img_path, cb)

            if mask is None or mask.sum() == 0:
                no_mask += 1
                done.add(img_path.name)
                continue

            boxes = mask_to_boxes(mask, sx, sy, sw, sh)
            if boxes:
                oname = f"{cls_name}_{img_path.stem}"
                opath = SAM_OUTPUT / "images" / cls_name / f"{oname}.jpg"
                # checkpoint 机制已保证不会重复处理，无需 exists() 检查
                ext = img_path.suffix.lower()
                if ext in ('.jpg', '.jpeg') and img.mode == 'RGB':
                    shutil.copy2(img_path, opath)
                else:
                    im = img
                    if im.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                        im = im.convert('RGB')
                    im.save(opath, quality=95)

                lpath = SAM_OUTPUT / "labels" / cls_name / f"{oname}.txt"
                with open(lpath, 'w') as f:
                    for b in boxes:
                        f.write(f"{cid} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

                total_boxes += len(boxes)
                total_with += 1
                ok += 1
            else:
                no_mask += 1

            done.add(img_path.name)

            if (idx + 1) % 200 == 0:
                ckpt = {"processed": list(done), "total_boxes": total_boxes,
                        "total_images": total_with,
                        "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_checkpoint(ckpt)
                el = time.time() - t0
                sp = (idx + 1) / el if el > 0 else 0
                eta = (len(remaining) - idx - 1) / sp if sp > 0 else 0
                avg = total_boxes / max(total_with, 1)
                print(f"  [{idx+1}/{len(remaining)}] ok={ok} no={no_mask} err={err} "
                      f"avg={avg:.1f}box sp={sp:.2f}/s ETA={eta/60:.0f}min")

        except Exception as e:
            err += 1
            if err <= 10:
                print(f"  [ERR] {img_path.name}: {e}")
            done.add(img_path.name)
            if err > 100:
                print("错误过多, 终止!")
                break

    ckpt = {"processed": list(done), "total_boxes": total_boxes,
            "total_images": total_with,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_checkpoint(ckpt)

    el = time.time() - t0
    print(f"\nSAM 完成! ok={ok} no_mask={no_mask} err={err} 耗时={el/60:.0f}min")
    print(f"总框={total_boxes} 平均={total_boxes/max(total_with,1):.1f}框/张")
    return ckpt


# ==================== Stage 2: 训练 ====================

def stage2_train():
    print("\n" + "=" * 60)
    print("STAGE 2: 切分数据集 + YOLOv26n 训练")
    print("=" * 60)

    # 收集 SAM 标注结果 (从类子目录)
    label_dir = SAM_OUTPUT / "labels"
    img_dir = SAM_OUTPUT / "images"
    labels = list(label_dir.glob("**/*.txt"))
    print(f"标注文件: {len(labels)}")

    # 按类别分组, 分层采样切分
    cls_groups = {i: [] for i in range(len(SELECTED_CLASSES))}
    for lp in labels:
        with open(lp, 'r') as f:
            line = f.readline().strip()
            if line:
                cid = int(line.split()[0])
                if cid in cls_groups:
                    cls_groups[cid].append(lp)

    train_files = []
    val_files = []
    for cid, lps in cls_groups.items():
        random.shuffle(lps)
        n_train = max(1, int(len(lps) * TRAIN_RATIO))
        train_files.extend(lps[:n_train])
        val_files.extend(lps[n_train:])
        print(f"  {SELECTED_CLASSES[cid]}: train={n_train} val={len(lps)-n_train}")

    print(f"\n总: train={len(train_files)} val={len(val_files)}")

    # 构建 YOLO 数据集目录
    TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, files in [("train", train_files), ("val", val_files)]:
        (TRAIN_OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (TRAIN_OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)
        for lp in files:
            stem = lp.stem
            cls_name = lp.parent.name
            src_img = img_dir / cls_name / f"{stem}.jpg"
            if src_img.exists():
                shutil.copy2(src_img, TRAIN_OUTPUT / split / "images" / f"{stem}.jpg")
            if lp.exists():
                shutil.copy2(lp, TRAIN_OUTPUT / split / "labels" / f"{stem}.txt")

    # data.yaml
    yaml = f"""path: {TRAIN_OUTPUT}
train: train/images
val: val/images
nc: {len(SELECTED_CLASSES)}
names: {SELECTED_CLASSES}
"""
    yaml_path = TRAIN_OUTPUT / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml)
    print(f"data.yaml → {yaml_path}")

    # ===== YOLOv26n 训练 =====
    print("\n开始训练 YOLOv26n...")
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMSZ,
        batch=BATCH,
        device=0,
        project=str(TRAIN_OUTPUT / "runs"),
        name="train_sam",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        close_mosaic=30,
        save=True, save_period=10,
        val=True,
        workers=4,
        pretrained=True, amp=True,
    )

    print("\n训练完成!")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")

    # 验证
    metrics = model.val(data=str(yaml_path), split="val", device=0)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return metrics


# ==================== Main ====================

def main():
    print("=" * 60)
    print("一条龙: SAM 标注 → YOLOv26n 训练")
    print(f"数据: {ARCHIVE_DIR}")
    print(f"SAM输出: {SAM_OUTPUT}")
    print(f"训练输出: {TRAIN_OUTPUT}")
    print(f"训练: {EPOCHS} epochs, {IMSZ}px, batch={BATCH}")
    print("=" * 60)

    # Stage 1
    stage1_sam()

    # Stage 2
    metrics = stage2_train()

    print("\n" + "=" * 60)
    print("一条龙完成!")
    print(f"最终 mAP50: {metrics.box.map50:.4f}")
    print(f"最终 mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
