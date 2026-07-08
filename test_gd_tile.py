"""
GD tiling pipeline 快速验证 —— 每类 1 张图，展示最终输出
"""
import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
sys.path.insert(0, r"D:\python\程序\mnist-YOLO\小麦病变识别模型")

import numpy as np
import cv2
from datetime import datetime
from annotate_gd_tile import *

# 每类挑 1 张
items = collect_images()
test = []
for cls_name in ["Black Rust", "Brown Rust", "Septoria", "Mildew"]:
    for img, cid, cn in items:
        if cn == cls_name:
            test.append((img, cid, cn))
            break

for img_path, cls_id, cls_name in test:
    print(f"\n{'='*70}")
    print(f"[{cls_id}] {cls_name}: {img_path.name}")

    with open(img_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_h, img_w = img_bgr.shape[:2]
    leaf_mask_arr = leaf_mask_bgr(img_bgr)

    tiles = get_tiles(img_h, img_w)
    valid_tiles = sum(1 for tx1, ty1, tx2, ty2 in tiles
                      if np.count_nonzero(leaf_mask_arr[ty1:ty2, tx1:tx2]) / ((tx2-tx1)*(ty2-ty1)) >= 0.05)

    t0 = datetime.now()
    disease_raw = tile_and_detect(img_bgr, CLASS_PROMPTS[cls_name], leaf_mask_arr)
    dt = (datetime.now() - t0).total_seconds()
    print(f"  Size={img_w}x{img_h}, tiles={len(tiles)}/{valid_tiles} valid, {dt:.1f}s")
    print(f"  Tile-filtered: {len(disease_raw)} boxes")

    disease_final = filter_boxes_global(disease_raw, img_w, img_h, leaf_mask_arr)
    print(f"  Global-filtered: {len(disease_final)} boxes")

    if disease_final:
        sorted_boxes = sorted(disease_final, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        print(f"  Smallest {min(5, len(sorted_boxes))}:")
        for b in sorted_boxes[:5]:
            bw, bh = b[2]-b[0], b[3]-b[1]
            print(f"    [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {bw:.0f}x{bh:.0f} ({bw*bh/(img_w*img_h)*100:.2f}%)")
        if len(sorted_boxes) > 5:
            print(f"  Largest 3:")
            for b in sorted_boxes[-3:]:
                bw, bh = b[2]-b[0], b[3]-b[1]
                print(f"    [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {bw:.0f}x{bh:.0f} ({bw*bh/(img_w*img_h)*100:.2f}%)")
    else:
        print(f"  NO BOXES after filtering!")

    yolo = boxes_to_yolo(disease_final, img_w, img_h, cls_id)
    print(f"  YOLO: {len(yolo)} lines")

print("\nDone!")
