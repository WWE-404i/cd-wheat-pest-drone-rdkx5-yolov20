"""
病灶框像素尺寸分析 — 按用户标准分级
  <12px: 直接剔除
  12-20px: 原图清晰可保留
  >=20px: 合格
"""
import os
from pathlib import Path
from PIL import Image
from collections import defaultdict

DATASET_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_yolo_train")
CLASS_NAMES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Mildew", "Fusarium Head Blight", "Healthy Wheat",
]

# 统计
box_stats = {i: {"lt12": 0, "mid": 0, "ge20": 0, "total": 0, "img_affected": set()}
             for i in range(7)}
total_images = 0
total_boxes = 0

# 遍历 train + val
for split in ["train", "val"]:
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

    if not lbl_dir.exists():
        continue

    for lbl_file in lbl_dir.glob("*.txt"):
        # 找对应图片
        img_file = img_dir / f"{lbl_file.stem}.jpg"
        if not img_file.exists():
            continue

        # 读图片尺寸
        try:
            with Image.open(img_file) as img:
                img_w, img_h = img.size
        except Exception:
            continue

        # 读标注
        lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue

        total_images += 1
        has_small = False

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            bw = float(parts[3]) * img_w  # 框像素宽度
            bh = float(parts[4]) * img_h  # 框像素高度
            min_side = min(bw, bh)

            box_stats[cls_id]["total"] += 1
            total_boxes += 1

            if min_side < 12:
                box_stats[cls_id]["lt12"] += 1
                has_small = True
            elif min_side < 20:
                box_stats[cls_id]["mid"] += 1
            else:
                box_stats[cls_id]["ge20"] += 1

        if has_small:
            box_stats[cls_id]["img_affected"].add(str(lbl_file))

# 打印
print("=" * 75)
print("病灶框像素尺寸统计")
print("=" * 75)
print(f"{'Class':<22} {'<12px删':>8} {'12-20px审':>10} {'>=20px留':>8} {'总框数':>8} {'受影响图':>8}")
print("-" * 75)

total_lt12 = total_mid = total_ge20 = 0
for cls_id in range(8):
    s = box_stats[cls_id]
    if s["total"] == 0:
        continue
    affected = len(s["img_affected"])
    print(f"{CLASS_NAMES[cls_id]:<22} {s['lt12']:>8} {s['mid']:>10} {s['ge20']:>8} {s['total']:>8} {affected:>8}")
    total_lt12 += s["lt12"]
    total_mid += s["mid"]
    total_ge20 += s["ge20"]

print("-" * 75)
all_boxes = total_lt12 + total_mid + total_ge20
print(f"{'合计':<22} {total_lt12:>8} {total_mid:>10} {total_ge20:>8} {all_boxes:>8}")
print()
if all_boxes > 0:
    print(f"<12px:   {total_lt12:>6} 框 ({total_lt12/all_boxes*100:5.1f}%) — 直接剔除")
    print(f"12-20px: {total_mid:>6} 框 ({total_mid/all_boxes*100:5.1f}%) — 原图清晰可保留")
    print(f">=20px:  {total_ge20:>6} 框 ({total_ge20/all_boxes*100:5.1f}%) — 合格")
print(f"\n总图片数: {total_images}")
