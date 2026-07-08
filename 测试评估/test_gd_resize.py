"""
测试 resize 策略：大图缩到短边 960，检查是否能检出小病变斑点
对比：direct / resize / tiling 三种方案
"""
import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
sys.path.insert(0, r"D:\python\程序\mnist-YOLO\小麦病变识别模型")

import numpy as np
import cv2
from datetime import datetime
from annotate_gd_tile import *
from annotate_gd_tile import _filter_tile_level

# 只测大图（>1920px）
items = collect_images()
test = []
for cls_name in ["Black Rust", "Brown Rust", "Septoria", "Mildew"]:
    for img, cid, cn in items:
        if cn == cls_name:
            pil = __import__('PIL').Image.open(img)
            if max(pil.size) > 1920:
                test.append((img, cid, cn))
                break

TARGET_SHORT = 960
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.10

def detect_resize(img_bgr, prompts, target_short=TARGET_SHORT):
    """Resize 大图使短边 = target_short，检测后映射回原始坐标"""
    img_h, img_w = img_bgr.shape[:2]
    short = min(img_h, img_w)

    if short <= target_short:
        # 不需要 resize
        return detect_on_tile(img_bgr, prompts), 1.0

    scale = target_short / short
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    t0 = datetime.now()
    boxes_resized = detect_on_tile(resized, prompts)
    dt = (datetime.now() - t0).total_seconds()

    # 映射回原始坐标
    boxes_original = []
    for box in boxes_resized:
        bx1, by1, bx2, by2 = box
        boxes_original.append([
            bx1 / scale, by1 / scale,
            bx2 / scale, by2 / scale
        ])

    return boxes_original, scale

def detect_tile(img_bgr, prompts, leaf_mask):
    t0 = datetime.now()
    boxes = tile_and_detect(img_bgr, prompts, leaf_mask)
    dt = (datetime.now() - t0).total_seconds()
    return boxes, dt

for img_path, cls_id, cls_name in test:
    print(f"\n{'='*70}")
    print(f"[{cls_id}] {cls_name}: {img_path.name}")

    with open(img_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_h, img_w = img_bgr.shape[:2]
    leaf_mask_arr = leaf_mask_bgr(img_bgr)

    prompts = CLASS_PROMPTS[cls_name]

    # === 方案1: Direct (no resize, no tile) ===
    t0 = datetime.now()
    direct_raw = detect_on_tile(img_bgr, prompts)
    dt_direct = (datetime.now() - t0).total_seconds()
    direct_filt = _filter_tile_level(direct_raw, img_w, img_h)
    print(f"  Direct:   {len(direct_raw)} raw -> {len(direct_filt)} filt ({dt_direct:.1f}s)")

    # === 方案2: Resize ===
    resize_raw, scale = detect_resize(img_bgr, prompts)
    resize_filt = _filter_tile_level(resize_raw, img_w, img_h)
    print(f"  Resize:   {len(resize_raw)} raw -> {len(resize_filt)} filt (scale={scale:.2f})")
    if resize_filt:
        sorted_b = sorted(resize_filt, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        print(f"    Smallest 3:")
        for b in sorted_b[:3]:
            bw, bh = b[2]-b[0], b[3]-b[1]
            print(f"      [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {bw:.0f}x{bh:.0f} ({bw*bh/(img_w*img_h)*100:.2f}%)")

    # === 方案3: Tiling (for comparison) ===
    tiles = get_tiles(img_h, img_w)
    print(f"  Tiling:   would need {len(tiles)} tiles (~{len(tiles)*0.3:.0f}s)")

    # After global filter
    resize_global = filter_boxes_global(resize_filt, img_w, img_h, leaf_mask_arr)
    print(f"  Resize + global filter: {len(resize_global)} boxes")
    if resize_global:
        sorted_b = sorted(resize_global, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        print(f"    Smallest 3:")
        for b in sorted_b[:3]:
            bw, bh = b[2]-b[0], b[3]-b[1]
            print(f"      [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {bw:.0f}x{bh:.0f} ({bw*bh/(img_w*img_h)*100:.2f}%)")

print("\nDone!")
