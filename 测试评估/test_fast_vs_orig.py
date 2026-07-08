"""
快速测试优化版 annotate_gd_tile_fast —— 3 张不同分辨率图
"""
import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
sys.path.insert(0, r"D:\python\程序\mnist-YOLO\小麦病变识别模型")

import numpy as np
import cv2
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
KEEP_DIR = PROJECT / "image_audit" / "keep_短边大于等于640"

# 找 3 张不同分辨率测试图
test_cases = []
for cls_name in ["Black Rust", "Brown Rust", "Septoria"]:
    cls_dir = KEEP_DIR / cls_name
    if not cls_dir.exists():
        continue
    for f in sorted(cls_dir.iterdir()):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            with open(f, 'rb') as fh:
                data = np.frombuffer(fh.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                test_cases.append((str(f), w, h, cls_name))
                break
test_cases.sort(key=lambda x: x[1] * x[2])
picks = [test_cases[len(test_cases)//4], test_cases[len(test_cases)//2], test_cases[-1]]

# 导入优化版（模型加载在模块顶层）
from annotate_gd_tile_fast import (
    leaf_mask_bgr, tile_and_detect, filter_boxes_global,
    CLASS_PROMPTS, TILE_SIZE, TILE_STRIDE,
)

print(f"\n{'='*70}")
print(f"优化版标注测试: {len(picks)} 张图")
print(f"Tile={TILE_SIZE}, Stride={TILE_STRIDE}")
print(f"{'='*70}")

for img_path, w, h, cls_name in picks:
    print(f"\n--- {cls_name}: {w}x{h} ---")

    with open(img_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # 计算 tiles
    tiles = []
    y = 0
    while y < h:
        if y + TILE_SIZE > h:
            y = max(0, h - TILE_SIZE)
        x = 0
        while x < w:
            if x + TILE_SIZE > w:
                x = max(0, w - TILE_SIZE)
            x2, y2 = min(x + TILE_SIZE, w), min(y + TILE_SIZE, h)
            tiles.append((x, y, x2, y2))
            if x2 >= w: break
            x += TILE_STRIDE
        if y2 >= h: break
        y += TILE_STRIDE
    tiles = list(set(tiles))

    leaf = leaf_mask_bgr(img_bgr)
    leaf_pct = np.count_nonzero(leaf) / leaf.size * 100

    prompts = CLASS_PROMPTS[cls_name]
    n_prompts = len(prompts)

    # 推理
    t0 = time.perf_counter()
    raw = tile_and_detect(img_bgr, prompts, leaf)
    final = filter_boxes_global(raw, w, h, leaf)
    dt = time.perf_counter() - t0

    print(f"  Tiles: {len(tiles)}, Prompts: {n_prompts}, Leaf: {leaf_pct:.0f}%")
    print(f"  Time: {dt:.2f}s, Raw: {len(raw)}, Final: {len(final)} boxes")

    if final:
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in final]
        print(f"  Areas: min={min(areas):.0f} max={max(areas):.0f} avg={np.mean(areas):.0f}px")

print(f"\n✅ 测试完成")
