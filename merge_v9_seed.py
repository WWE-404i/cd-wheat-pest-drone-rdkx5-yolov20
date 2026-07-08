"""
V9种子: 精标+审核全部726张, 锈病(0/1/2)用终审标签, 非锈病保留
从零训练, 用于第一轮伪标迭代
"""
from pathlib import Path
import shutil
from collections import Counter

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN = PROJECT / "wheat_disease_golden"
REVIEWED = PROJECT / "wheat_pseudo_v5_reviewed"
RUST_REVIEW = PROJECT / "rust_review"
OUT = PROJECT / "wheat_v9_seed"
SRC_IMAGES = PROJECT / "wheat_disease_8class"

CLS_NAMES = ['Brown_Rust','Yellow_Rust','Black_Rust','Septoria',
             'Powdery_Mildew','Fusarium_HB','Healthy']
RUST_CLASSES = {0, 1, 2}

if OUT.exists():
    shutil.rmtree(OUT)
for split in ["train", "val"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

def find_image(stem):
    for s in ["train", "val"]:
        p = SRC_IMAGES / s / "images" / f"{stem}.jpg"
        if p.exists(): return p
    for s in ["train", "val"]:
        p = GOLDEN / s / "images" / f"{stem}.jpg"
        if p.exists(): return p
    return None

def parse_label(path):
    boxes = []
    if not path.exists(): return boxes
    for line in path.read_text().strip().splitlines():
        if not line.strip(): continue
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes

# 锈病终审索引
rust_labels = {}
if (RUST_REVIEW / "labels").exists():
    for lbl in (RUST_REVIEW / "labels").glob("*.txt"):
        boxes = parse_label(lbl)
        rust_labels[lbl.stem] = boxes
print(f"锈病终审: {len(rust_labels)} 张")

seen = set()
counts = {"golden": 0, "reviewed": 0}
rust_replaced = 0
missing = 0

# === 精标 ===
for split in ["train", "val"]:
    img_dir = GOLDEN / split / "images"
    lbl_dir = GOLDEN / split / "labels"
    if not img_dir.exists(): continue
    for img in img_dir.glob("*.jpg"):
        stem = img.stem
        if stem in seen: continue
        seen.add(stem)
        counts["golden"] += 1

        if stem in rust_labels:
            # 锈病终审替换该类，非锈病保留原标签
            old_boxes = parse_label(lbl_dir / f"{stem}.txt")
            rust_boxes = rust_labels[stem]
            # 替换: 去掉旧的锈病框, 加上终审的
            nonrust = [b for b in old_boxes if b[0] not in RUST_CLASSES]
            final_boxes = nonrust + rust_boxes
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in final_boxes]
            dst_lbl = OUT / split / "labels" / f"{stem}.txt"
            dst_lbl.write_text("\n".join(lines))
            rust_replaced += 1
        else:
            src_lbl = lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, OUT / split / "labels" / f"{stem}.txt")
            else:
                missing += 1; continue

        shutil.copy2(img, OUT / split / "images" / img.name)

# === 审核 ===
rev_lbl_dir = REVIEWED / "labels"
if rev_lbl_dir.exists():
    for lbl in rev_lbl_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in seen: continue
        seen.add(stem)
        counts["reviewed"] += 1

        img = find_image(stem)
        if img is None: missing += 1; continue

        if stem in rust_labels:
            old_boxes = parse_label(lbl)
            rust_boxes = rust_labels[stem]
            nonrust = [b for b in old_boxes if b[0] not in RUST_CLASSES]
            final_boxes = nonrust + rust_boxes
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in final_boxes]
            (OUT / "train" / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            rust_replaced += 1
        else:
            shutil.copy2(lbl, OUT / "train" / "labels" / f"{stem}.txt")

        shutil.copy2(img, OUT / "train" / "images" / img.name)

print(f"精标: {counts['golden']} | 审核: {counts['reviewed']} | 合计: {sum(counts.values())}")
print(f"锈病替换: {rust_replaced} | 缺图: {missing}")

# 统计
total = 0
cls_cnt = Counter()
for split in ["train", "val"]:
    ld = OUT / split / "labels"
    if not ld.exists(): continue
    for f in ld.glob("*.txt"):
        total += 1
        for line in f.read_text().strip().splitlines():
            if line.strip():
                cls_cnt[int(line.split()[0])] += 1

print(f"\n总图: {total}")
for c in range(7):
    print(f"  [{c}] {CLS_NAMES[c]}: {cls_cnt.get(c,0)}")
print(f"\n→ {OUT}")
