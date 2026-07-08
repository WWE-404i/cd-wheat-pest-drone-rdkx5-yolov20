"""
快速验证 GD v3 —— 5 张图 (box=0.25, text=0.20)
"""
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
from annotate_gd_v3 import *

# 只取前 5 张非健康图片
items = collect_images()
test_items = []
seen_cls = set()
for img, cid, cname in items:
    if cname != "Healthy Wheat" and cname not in seen_cls:
        test_items.append((img, cid, cname))
        seen_cls.add(cname)
for img, cid, cname in items:
    if cname == "Healthy Wheat" and "Healthy Wheat" not in [c for _, _, c in test_items]:
        test_items.append((img, cid, cname))
        break
test_items = test_items[:6]

print(f"\nTest images ({len(test_items)}):")
for img, cid, cname in test_items:
    print(f"  [{cid}] {cname}: {img.name}")

for img_path, cls_id, cls_name in test_items:
    print(f"\n{'='*60}")
    print(f"[{cls_id}] {cls_name}: {img_path.name}")

    with open(img_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_h, img_w = img_bgr.shape[:2]
    leaf_mask_arr = leaf_mask_bgr(img_bgr)
    print(f"  Size: {img_w}x{img_h}")

    # 临时用低阈值: box=0.15, text=0.15
    disease_raw = detect_with_prompts(img_bgr, CLASS_PROMPTS[cls_name], box_threshold=0.15, text_threshold=0.15)
    # 展示所有 raw box（不过滤）
    print(f"  Disease raw (all, no filter): {len(disease_raw)}")
    # 按面积排序，展示最小5个和最大3个
    sorted_boxes = sorted(disease_raw, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    print(f"  Smallest 5:")
    for b in sorted_boxes[:5]:
        bw, bh = b[2]-b[0], b[3]-b[1]
        print(f"    [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {bw:.0f}x{bh:.0f} area={bw*bh:.0f}")
    print(f"  Largest 3:")
    for b in sorted_boxes[-3:]:
        bw, bh = b[2]-b[0], b[3]-b[1]
        print(f"    [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {bw:.0f}x{bh:.0f} area={bw*bh:.0f}")

    # 健康检测
    healthy_raw = detect_with_prompts(img_bgr, CLASS_PROMPTS["Healthy Wheat"], box_threshold=0.15, text_threshold=0.15)
    print(f"  Healthy raw: {len(healthy_raw)} total")
    leaf_pct = np.count_nonzero(leaf_mask_arr) / leaf_mask_arr.size * 100
    print(f"  Leaf coverage: {leaf_pct:.1f}%")

print("\nDone!")
