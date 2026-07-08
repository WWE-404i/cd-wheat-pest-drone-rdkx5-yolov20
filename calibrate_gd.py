"""
GD 阈值校准 — 用黄金标注集找到最佳 box_threshold + text_threshold
输出: best_thresholds.json
"""
import json
import time
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# ========== 配置 ==========
GOLDEN_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\golden_set")
LABEL_DIR = GOLDEN_DIR / "labels"
IMAGE_DIR = GOLDEN_DIR / "images"
OUTPUT_FILE = GOLDEN_DIR / "best_thresholds.json"

CLASS_NAMES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Mildew", "Fusarium Head Blight", "Healthy Wheat",
]

# GD 模型
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# 统一检测 prompt (只定位病害)
DETECTION_PROMPT = (
    "rust spots on wheat leaf . "
    "yellow stripes on wheat leaf . "
    "dark lesions on wheat . "
    "leaf blotch with yellow halo . "
    "powdery white patches on wheat . "
    "bleached wheat spike head . "
    "dead spots blight on leaf"
)

# 搜索空间
BOX_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
TEXT_THRESHOLDS = [0.15, 0.20, 0.25, 0.30]
IOU_MATCH = 0.5   # IoU > 0.5 算匹配成功
MIN_BOX_AREA = 0.005
MAX_BOX_AREA = 0.85


# ============================================================
def load_model():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def detect(model, processor, img_path, box_thr, text_thr):
    """GD 检测, 返回 YOLO 归一化框列表 [(cx,cy,w,h), ...]"""
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


def load_human_labels():
    """加载黄金标注: {img_stem: [(cls_id, cx,cy,w,h), ...]}"""
    human = {}
    for lbl_file in LABEL_DIR.glob("*.txt"):
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


def compute_iou(b1, b2):
    """两个 YOLO 归一化框的 IoU"""
    # b = (cx, cy, w, h)
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


def evaluate(gd_boxes, human_boxes):
    """
    计算 precision/recall/F1
    gd_boxes: [(cls, cx,cy,w,h), ...] — GD检测 + 文件夹类别
    human_boxes: [(cls, cx,cy,w,h), ...]
    """
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
            if h_cls != g_cls:
                continue
            if hi in matched_human:
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


def main():
    print("=" * 60)
    print("GD 阈值校准")
    print(f"黄金集: {GOLDEN_DIR}")
    print(f"搜索空间: box={BOX_THRESHOLDS}, text={TEXT_THRESHOLDS}")
    print("=" * 60)

    # 加载黄金标注
    human_labels = load_human_labels()
    print(f"\n黄金标注: {len(human_labels)} 张")

    # 统计黄金集
    total_human_boxes = sum(len(v) for v in human_labels.values())
    images_with_boxes = sum(1 for v in human_labels.values() if v)
    print(f"  有框图片: {images_with_boxes}, 总框: {total_human_boxes}")

    # 加载 GD 模型
    model, processor = load_model()

    # 网格搜索
    results = []
    total_combos = len(BOX_THRESHOLDS) * len(TEXT_THRESHOLDS)
    combo_idx = 0

    for box_thr in BOX_THRESHOLDS:
        for text_thr in TEXT_THRESHOLDS:
            combo_idx += 1
            agg = {"tp": 0, "fp": 0, "fn": 0}

            t0 = time.time()
            for lbl_file in LABEL_DIR.glob("*.txt"):
                stem = lbl_file.stem
                img_path = IMAGE_DIR / f"{stem}.jpg"
                if not img_path.exists():
                    img_path = IMAGE_DIR / f"{stem}.png"
                if not img_path.exists():
                    continue

                # 从文件名提取类别 (用于GD框的类别赋值)
                fn_cls_name = stem.split('_')[0]
                cls_id = CLASS_NAMES.index(fn_cls_name) if fn_cls_name in CLASS_NAMES else -1

                # Healthy Wheat: 跳过GD, 直接评估
                if cls_id == 6:
                    gd_boxes_raw = []
                else:
                    gd_boxes_raw = detect(model, processor, img_path, box_thr, text_thr)

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
            print(f"  [{combo_idx}/{total_combos}] box={box_thr:.2f} text={text_thr:.2f} | "
                  f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} | {elapsed:.0f}s")

            results.append({
                "box_threshold": box_thr,
                "text_threshold": text_thr,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            })

    # 找最佳
    results.sort(key=lambda x: x["f1"], reverse=True)
    best = results[0]

    print(f"\n{'='*60}")
    print(f"最佳阈值: box={best['box_threshold']}, text={best['text_threshold']}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  Recall:    {best['recall']:.3f}")
    print(f"  F1:        {best['f1']:.3f}")
    print(f"  TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    print(f"{'='*60}")

    # 保存所有结果
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            "best": best,
            "all_results": results,
            "golden_stats": {
                "images": len(human_labels),
                "images_with_boxes": images_with_boxes,
                "total_boxes": total_human_boxes,
            }
        }, f, indent=2)

    print(f"\n结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
