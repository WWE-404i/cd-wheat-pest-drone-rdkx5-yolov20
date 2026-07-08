"""
一条龙自动化流水线: 校准 → 标注 → 过滤 → 训练
睡前运行, 醒来收结果

流程:
  Phase 1: GD 阈值校准 (黄金集 128 张)
  Phase 2: GD 全量标注 (13K+ 图片, 7 类)
  Phase 3: 图片分辨率过滤 + 框尺寸过滤
  Phase 4: YOLOv26n 训练
"""

import os
import json
import time
import random
import shutil
import subprocess
import sys
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import torch
import numpy as np

# ============================================================
# 全局配置
# ============================================================
PROJECT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN_DIR = PROJECT_DIR / "golden_set"
ARCHIVE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\archive\Wheat_Disease")
OUTPUT_DIR = PROJECT_DIR / "wheat_yolo_train"
LOG_FILE = PROJECT_DIR / "pipeline_full.log"

CLASS_NAMES = [
    "Brown Rust",           # 0
    "Yellow Rust",          # 1
    "Black Rust",           # 2
    "Septoria",             # 3
    "Mildew",               # 4
    "Fusarium Head Blight", # 5
    "Healthy Wheat",        # 6
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# GD 参数
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DETECTION_PROMPT = (
    "rust spots on wheat leaf . "
    "yellow stripes on wheat leaf . "
    "dark lesions on wheat . "
    "leaf blotch with yellow halo . "
    "powdery white patches on wheat . "
    "bleached wheat spike head . "
    "dead spots blight on leaf"
)

# 校准搜索空间
BOX_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
TEXT_THRESHOLDS = [0.15, 0.20, 0.25, 0.30]
IOU_MATCH = 0.5
MIN_BOX_AREA = 0.005
MAX_BOX_AREA = 0.85

# 数据集切分
VAL_RATIO = 0.15
SEED = 42

# 训练参数
TRAIN_EPOCHS = 200
TRAIN_IMSZ = 640
TRAIN_BATCH = 40
TRAIN_DEVICE = 0

# 分辨率过滤阈值
MIN_SHORT_SIDE_DELETE = 480   # 短边 < 480 直接删
MIN_SHORT_SIDE_REVIEW = 640   # 480 ~ 640 人工复核

# 框尺寸过滤阈值
BOX_MIN_PX_DELETE = 12   # 宽/高 < 12px 直接删
BOX_MIN_PX_REVIEW = 20   # 12 ~ 20px 标记

# ============================================================
# 日志
# ============================================================
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ============================================================
# Phase 1: GD 阈值校准
# ============================================================

def load_gd_model():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    t0 = time.time()
    log(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to("cuda").eval()
    vram = torch.cuda.memory_allocated() / 1024**3
    log(f"GD 模型就绪. {time.time()-t0:.1f}s, VRAM={vram:.1f}GB")
    return model, processor


def gd_detect(model, processor, img_path, box_thr, text_thr):
    """GD 检测 → YOLO 归一化框 [(cx,cy,w,h), ...]"""
    img = Image.open(img_path).convert("RGB")
    ow, oh = img.size
    inputs = processor(images=img, text=DETECTION_PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_thr, text_threshold=text_thr,
        target_sizes=[(oh, ow)]
    )[0]
    boxes = results["boxes"].cpu().numpy()
    yolo = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = max(0.0, min(1.0, (x1+x2)/2/ow))
        cy = max(0.0, min(1.0, (y1+y2)/2/oh))
        w = max(0.005, min(1.0, (x2-x1)/ow))
        h = max(0.005, min(1.0, (y2-y1)/oh))
        if MIN_BOX_AREA <= w*h <= MAX_BOX_AREA:
            yolo.append((cx, cy, w, h))
    return yolo


def compute_iou(b1, b2):
    """YOLO 归一化框 IoU"""
    x1_a, y1_a = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    x2_a, y2_a = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    x1_b, y1_b = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    x2_b, y2_b = b2[0]+b2[2]/2, b2[1]+b2[3]/2
    inter_x1, inter_y1 = max(x1_a, x1_b), max(y1_a, y1_b)
    inter_x2, inter_y2 = min(x2_a, x2_b), min(y2_a, y2_b)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2-inter_x1) * (inter_y2-inter_y1)
    area_a = (x2_a-x1_a) * (y2_a-y1_a)
    area_b = (x2_b-x1_b) * (y2_b-y1_b)
    return inter / (area_a + area_b - inter + 1e-8)


def load_human_labels():
    """加载黄金标注 {stem: [(cls, cx,cy,w,h), ...]}"""
    label_dir = GOLDEN_DIR / "labels"
    human = {}
    for lbl_file in label_dir.glob("*.txt"):
        stem = lbl_file.stem
        boxes = []
        for line in lbl_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls, cx, cy, w, h))
        human[stem] = boxes
    return human


def evaluate(gd_boxes, human_boxes):
    """计算 precision/recall/F1"""
    if not human_boxes:
        return {"tp": 0, "fp": len(gd_boxes), "fn": 0,
                "precision": 0.0, "recall": 1.0 if not gd_boxes else 0.0, "f1": 0.0}
    if not gd_boxes:
        return {"tp": 0, "fp": 0, "fn": len(human_boxes),
                "precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_human = set()
    matched_gd = set()
    for gi, (g_cls, gx, gy, gw, gh) in enumerate(gd_boxes):
        best_iou = 0
        best_hi = -1
        for hi, (h_cls, hx, hy, hw, hh) in enumerate(human_boxes):
            if h_cls != g_cls or hi in matched_human:
                continue
            iou = compute_iou((gx, gy, gw, gh), (hx, hy, hw, hh))
            if iou > best_iou:
                best_iou = iou
                best_hi = hi
        if best_iou >= IOU_MATCH and best_hi >= 0:
            matched_human.add(best_hi)
            matched_gd.add(gi)

    tp = len(matched_gd)
    fp = len(gd_boxes) - tp
    fn = len(human_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def phase1_calibrate(model, processor):
    """Phase 1: GD 阈值校准"""
    log("=" * 60)
    log("Phase 1: GD 阈值校准")
    log("=" * 60)

    human_labels = load_human_labels()
    log(f"黄金标注: {len(human_labels)} 张图")

    total_human_boxes = sum(len(v) for v in human_labels.values())
    images_with_boxes = sum(1 for v in human_labels.values() if v)
    log(f"  有框: {images_with_boxes} 张, 总框: {total_human_boxes}")

    # 建立 stem → class_id 映射
    stem_cls = {}
    for lbl_file in (GOLDEN_DIR / "labels").glob("*.txt"):
        stem = lbl_file.stem
        for cls_name in CLASS_NAMES:
            if stem.startswith(cls_name + "_"):
                stem_cls[stem] = CLASS_NAMES.index(cls_name)
                break

    log(f"搜索空间: box={BOX_THRESHOLDS}, text={TEXT_THRESHOLDS}")
    log(f"共 {len(BOX_THRESHOLDS) * len(TEXT_THRESHOLDS)} 组")

    results = []
    total_combos = len(BOX_THRESHOLDS) * len(TEXT_THRESHOLDS)
    combo_idx = 0
    t_start = time.time()

    for box_thr in BOX_THRESHOLDS:
        for text_thr in TEXT_THRESHOLDS:
            combo_idx += 1
            agg = {"tp": 0, "fp": 0, "fn": 0}
            t0 = time.time()

            for lbl_file in (GOLDEN_DIR / "labels").glob("*.txt"):
                stem = lbl_file.stem
                img_path = GOLDEN_DIR / "images" / f"{stem}.jpg"
                if not img_path.exists():
                    img_path = GOLDEN_DIR / "images" / f"{stem}.png"
                if not img_path.exists():
                    continue

                cls_id = stem_cls.get(stem, -1)
                if cls_id == 6:  # Healthy: 不跑 GD
                    gd_boxes_raw = []
                else:
                    gd_boxes_raw = gd_detect(model, processor, img_path, box_thr, text_thr)

                gd_boxes = [(cls_id, cx, cy, w, h) for cx, cy, w, h in gd_boxes_raw]
                h_boxes = human_labels.get(stem, [])
                scores = evaluate(gd_boxes, h_boxes)
                agg["tp"] += scores["tp"]
                agg["fp"] += scores["fp"]
                agg["fn"] += scores["fn"]

            tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
            prec = tp / (tp+fp) if (tp+fp) > 0 else 0.0
            rec = tp / (tp+fn) if (tp+fn) > 0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

            elapsed = time.time() - t0
            log(f"  [{combo_idx}/{total_combos}] box={box_thr:.2f} text={text_thr:.2f} | "
                f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} | {elapsed:.0f}s")

            results.append({
                "box_threshold": box_thr,
                "text_threshold": text_thr,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            })

    results.sort(key=lambda x: x["f1"], reverse=True)
    best = results[0]

    log(f"\n{'='*60}")
    log(f"✅ 最佳阈值: box={best['box_threshold']}, text={best['text_threshold']}")
    log(f"   Precision: {best['precision']:.3f}")
    log(f"   Recall:    {best['recall']:.3f}")
    log(f"   F1:        {best['f1']:.3f}")
    log(f"   TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    log(f"{'='*60}")

    # 保存
    out_json = GOLDEN_DIR / "best_thresholds.json"
    with open(out_json, 'w') as f:
        json.dump({
            "best": best,
            "all_results": results,
            "golden_stats": {
                "images": len(human_labels),
                "images_with_boxes": images_with_boxes,
                "total_boxes": total_human_boxes,
            }
        }, f, indent=2)
    log(f"阈值已保存: {out_json}")

    return best["box_threshold"], best["text_threshold"]


# ============================================================
# Phase 2: GD 全量标注
# ============================================================

def _nms(boxes, iou_threshold):
    """YOLO 格式 NMS"""
    items = [(b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2, b) for b in boxes]
    items.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
    kept = []
    while items:
        bx1, by1, bx2, by2, b = items.pop(0)
        kept.append(b)
        b_area = (bx2 - bx1) * (by2 - by1)
        filtered = []
        for ix1, iy1, ix2, iy2, ib in items:
            inter_x1, inter_y1 = max(bx1, ix1), max(by1, iy1)
            inter_x2, inter_y2 = min(bx2, ix2), min(by2, iy2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                iou = inter / (b_area + (ix2-ix1)*(iy2-iy1) - inter + 1e-8)
                if iou < iou_threshold:
                    filtered.append((ix1, iy1, ix2, iy2, ib))
            else:
                filtered.append((ix1, iy1, ix2, iy2, ib))
        items = filtered
    return kept


def collect_all_images():
    """收集所有图片 [(path, class_name), ...]"""
    all_imgs = []
    for cls_name in CLASS_NAMES:
        d = ARCHIVE_DIR / "train" / cls_name
        if d.exists():
            for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
                for p in d.glob(ext):
                    all_imgs.append((p, cls_name))
    return all_imgs


def make_split(all_images):
    """按类别分层采样 → {img_name: 'train'|'val'}"""
    random.seed(SEED)
    split_map = {}
    for cls_name in CLASS_NAMES:
        imgs = [(p, c) for p, c in all_images if c == cls_name]
        seen = set()
        unique = []
        for p, c in imgs:
            if p.name not in seen:
                seen.add(p.name)
                unique.append((p, c))
        random.shuffle(unique)
        n_val = max(1, int(len(unique) * VAL_RATIO))
        for i, (p, _) in enumerate(unique):
            split_map[p.name] = "val" if i < n_val else "train"
    return split_map


def phase2_annotate(model, processor, box_thr, text_thr):
    """Phase 2: GD 全量标注"""
    log("=" * 60)
    log("Phase 2: GD 全量标注")
    log(f"阈值: box={box_thr:.2f}, text={text_thr:.2f}")
    log("=" * 60)

    all_images = collect_all_images()
    log(f"总图片: {len(all_images)}")

    for cls, cnt in Counter(c for _, c in all_images).most_common():
        log(f"  {cls}: {cnt}")

    # train/val 切分
    log("\n确定 train/val 切分...")
    split_map = make_split(all_images)
    train_n = sum(1 for v in split_map.values() if v == "train")
    val_n = sum(1 for v in split_map.values() if v == "val")
    log(f"  Train: {train_n}, Val: {val_n}")

    # 清理 + 创建输出目录
    if OUTPUT_DIR.exists():
        log(f"清理旧输出: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # 断点续传
    checkpoint_file = OUTPUT_DIR / "gd_annotate_checkpoint.json"
    processed_set = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        processed_set = set(ckpt.get("processed", []))
        log(f"从断点续传: 已处理 {len(processed_set)} 张")

    remaining = [(p, c) for p, c in all_images if p.name not in processed_set]
    if not remaining:
        log("全部已处理!")
        return

    log(f"剩余: {len(remaining)} 张")

    total_boxes = 0
    total_with = 0
    total_processed = len(processed_set)
    ok = no_det = err = 0
    t0 = time.time()

    for idx, (img_path, folder_class) in enumerate(remaining):
        class_id = CLASS_TO_ID[folder_class]
        fname = img_path.name

        try:
            if class_id == 6:  # Healthy Wheat
                yolo_boxes = []
                no_det += 1
            else:
                raw = gd_detect(model, processor, img_path, box_thr, text_thr)
                if len(raw) > 1:
                    raw = _nms(raw, 0.6)
                yolo_boxes = [[cx, cy, w, h] for cx, cy, w, h in raw]
                if yolo_boxes:
                    total_boxes += len(yolo_boxes)
                    total_with += 1
                    ok += 1
                else:
                    no_det += 1

            # 保存
            split = split_map[fname]
            stem = img_path.stem
            out_name = f"{folder_class}_{stem}"
            img_dir = OUTPUT_DIR / split / "images"
            lbl_dir = OUTPUT_DIR / split / "labels"

            out_img = img_dir / f"{out_name}.jpg"
            if not out_img.exists():
                img = Image.open(img_path)
                if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                    img = img.convert('RGB')
                img.save(out_img, quality=95)

            out_lbl = lbl_dir / f"{out_name}.txt"
            with open(out_lbl, 'w') as f:
                for box in yolo_boxes:
                    f.write(f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

            processed_set.add(fname)
            total_processed += 1

        except Exception as e:
            err += 1
            if err <= 10:
                log(f"  [ERROR] {fname}: {e}")
            processed_set.add(fname)
            total_processed += 1
            if err > 100:
                log("❌ 错误过多, 终止!")
                break

        # 每 200 张保存进度
        if (idx + 1) % 200 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "processed": list(processed_set),
                    "total_boxes": total_boxes,
                    "total_images": total_with,
                    "total_processed": total_processed,
                }, f)
            elapsed = time.time() - t0
            spd = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - idx - 1) / spd if spd > 0 else 0
            pct = total_with / max(total_processed, 1) * 100
            log(f"  [{idx+1}/{len(remaining)}] ok={ok} no={no_det} err={err} "
                f"有框率={pct:.1f}% spd={spd:.1f}/s ETA={eta/60:.0f}min")

    # 最终保存
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "processed": list(processed_set),
            "total_boxes": total_boxes,
            "total_images": total_with,
            "total_processed": total_processed,
        }, f)

    elapsed = time.time() - t0
    log(f"\n{'='*60}")
    log(f"GD 标注完成!")
    log(f"  检测到框: {ok} 张 ({ok/max(len(remaining),1)*100:.1f}%)")
    log(f"  未检测到: {no_det} 张")
    log(f"  出错: {err} 张")
    log(f"  总框: {total_boxes} | 平均: {total_boxes/max(total_with,1):.1f} 框/有框图")
    log(f"  耗时: {elapsed/60:.1f} min")
    log(f"{'='*60}")

    # data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n",
        encoding='utf-8'
    )
    log(f"data.yaml → {yaml_path}")


# ============================================================
# Phase 3: 后处理过滤
# ============================================================

def phase3_filter():
    """Phase 3: 分辨率 + 框尺寸过滤"""
    log("=" * 60)
    log("Phase 3: 后处理过滤")
    log("=" * 60)

    # 统计
    deleted_images = []
    review_images = []
    deleted_boxes = []
    review_boxes = []

    for split in ["train", "val"]:
        img_dir = OUTPUT_DIR / split / "images"
        lbl_dir = OUTPUT_DIR / split / "labels"
        if not img_dir.exists():
            continue

        for lbl_file in list(lbl_dir.glob("*.txt")):
            stem = lbl_file.stem
            img_file = img_dir / f"{stem}.jpg"
            if not img_file.exists():
                continue

            # 读图片尺寸
            try:
                with Image.open(img_file) as im:
                    w, h = im.size
                short_side = min(w, h)
            except Exception:
                continue

            # 分辨率过滤
            if short_side < MIN_SHORT_SIDE_DELETE:
                deleted_images.append((str(img_file), short_side))
                img_file.unlink()
                lbl_file.unlink()
                continue
            elif short_side < MIN_SHORT_SIDE_REVIEW:
                review_images.append((str(img_file), short_side))

            # 框尺寸过滤
            lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
            if not lines:
                continue

            kept_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    kept_lines.append(line)
                    continue
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                min_side = min(bw, bh)

                if min_side < BOX_MIN_PX_DELETE:
                    deleted_boxes.append((str(img_file), min_side, line))
                    # 删除这个框
                    continue
                elif min_side < BOX_MIN_PX_REVIEW:
                    review_boxes.append((str(img_file), min_side, line))

                kept_lines.append(line)

            # 写回
            if len(kept_lines) != len(lines):
                if kept_lines:
                    lbl_file.write_text('\n'.join(kept_lines) + '\n')
                else:
                    lbl_file.write_text('')

    # 报告
    log(f"\n图片过滤 (短边 < {MIN_SHORT_SIDE_DELETE}px 删除): {len(deleted_images)} 张")
    if deleted_images:
        # 存到审计目录
        audit_dir = PROJECT_DIR / "image_audit" / "to_delete_pipeline"
        audit_dir.mkdir(parents=True, exist_ok=True)
        log(f"  已删除, 列表存至: {audit_dir}")
        with open(audit_dir / "deleted_list.txt", 'w') as f:
            for path, ss in deleted_images:
                f.write(f"{path}  ({ss}px)\n")

    log(f"图片待复核 (短边 {MIN_SHORT_SIDE_DELETE}~{MIN_SHORT_SIDE_REVIEW}px): {len(review_images)} 张")
    if review_images:
        review_img_dir = PROJECT_DIR / "image_audit" / "manual_review"
        review_img_dir.mkdir(parents=True, exist_ok=True)
        log(f"  列表存至: {review_img_dir}")
        with open(review_img_dir / "review_list_pipeline.txt", 'w') as f:
            for path, ss in review_images:
                f.write(f"{path}  ({ss}px)\n")

    log(f"\n框过滤 (宽/高 < {BOX_MIN_PX_DELETE}px 删除): {len(deleted_boxes)} 个框")
    log(f"框待复核 (宽/高 {BOX_MIN_PX_DELETE}~{BOX_MIN_PX_REVIEW}px): {len(review_boxes)} 个框")

    # 统计最终数据集
    train_imgs = len(list((OUTPUT_DIR / "train" / "images").glob("*.jpg")))
    val_imgs = len(list((OUTPUT_DIR / "val" / "images").glob("*.jpg")))
    train_labels = len(list((OUTPUT_DIR / "train" / "labels").glob("*.txt")))
    val_labels = len(list((OUTPUT_DIR / "val" / "labels").glob("*.txt")))

    log(f"\n最终数据集:")
    log(f"  Train: {train_imgs} 图片, {train_labels} 标注")
    log(f"  Val:   {val_imgs} 图片, {val_labels} 标注")
    log(f"  Total: {train_imgs + val_imgs} 图片")

    # 统计每类框数
    all_boxes_by_cls = Counter()
    for split in ["train", "val"]:
        for lbl_file in (OUTPUT_DIR / split / "labels").glob("*.txt"):
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    all_boxes_by_cls[int(parts[0])] += 1

    log("\n各类框数:")
    for cls_id in range(len(CLASS_NAMES)):
        log(f"  [{cls_id}] {CLASS_NAMES[cls_id]}: {all_boxes_by_cls.get(cls_id, 0)}")
    log(f"  总框: {sum(all_boxes_by_cls.values())}")


# ============================================================
# Phase 4: 训练
# ============================================================

def phase4_train():
    """Phase 4: YOLOv26n 训练"""
    log("=" * 60)
    log("Phase 4: YOLOv26n 训练")
    log("=" * 60)

    from ultralytics import YOLO

    data_yaml = OUTPUT_DIR / "data.yaml"
    if not data_yaml.exists():
        log("❌ data.yaml 不存在, 跳过训练")
        return

    model = YOLO("yolo26n.pt")
    project = str(PROJECT_DIR / "runs" / "wheat_baseline")
    name = "train_auto"

    log(f"数据: {data_yaml}")
    log(f"Epochs: {TRAIN_EPOCHS}, ImgSz: {TRAIN_IMSZ}, Batch: {TRAIN_BATCH}")
    log(f"输出: {project}/{name}")

    results = model.train(
        data=str(data_yaml),
        epochs=TRAIN_EPOCHS,
        imgsz=TRAIN_IMSZ,
        batch=TRAIN_BATCH,
        device=TRAIN_DEVICE,
        project=project,
        name=name,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        close_mosaic=30,
        save=True, save_period=10,
        val=True,
        workers=0,
        pretrained=True,
        amp=True,
        verbose=True,
    )

    log(f"\n训练完成!")
    log(f"最佳模型: {results.save_dir}/weights/best.pt")

    # 验证
    log("\n运行验证...")
    metrics = model.val(data=str(data_yaml), split="val", device=TRAIN_DEVICE)
    log(f"mAP50:    {metrics.box.map50:.4f}")
    log(f"mAP50-95: {metrics.box.map:.4f}")

    return results


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 60)
    log("🚀 一条龙自动化流水线启动")
    log(f"项目目录: {PROJECT_DIR}")
    log(f"日志文件: {LOG_FILE}")
    log("=" * 60)

    total_start = time.time()

    # Phase 1: 校准
    try:
        model, processor = load_gd_model()
        best_box, best_text = phase1_calibrate(model, processor)
    except Exception as e:
        log(f"❌ Phase 1 失败: {e}")
        log("使用默认阈值: box=0.30, text=0.25")
        best_box, best_text = 0.30, 0.25
        # 尝试加载模型继续
        try:
            model, processor = load_gd_model()
        except Exception:
            log("❌ 无法加载 GD 模型, 跳过标注阶段")
            model, processor = None, None

    # Phase 2: 全量标注
    if model is not None and processor is not None:
        try:
            phase2_annotate(model, processor, best_box, best_text)
            # 释放 GD 模型显存
            del model, processor
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"❌ Phase 2 失败: {e}")
            import traceback
            log(traceback.format_exc())
    else:
        log("跳过 Phase 2 (无 GD 模型)")

    # Phase 3: 过滤
    try:
        phase3_filter()
    except Exception as e:
        log(f"❌ Phase 3 失败: {e}")
        import traceback
        log(traceback.format_exc())

    # Phase 4: 训练
    try:
        phase4_train()
    except Exception as e:
        log(f"❌ Phase 4 失败: {e}")
        import traceback
        log(traceback.format_exc())

    total_elapsed = time.time() - total_start
    log(f"\n{'='*60}")
    log(f"🎉 全流程完成! 总耗时: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f}h)")
    log(f"日志文件: {LOG_FILE}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
