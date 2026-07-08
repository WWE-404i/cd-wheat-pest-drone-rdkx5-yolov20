"""
V7: 精标+审核+锈病终审 = 纯净726张
"""
from pathlib import Path
import shutil
from collections import Counter

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN = PROJECT / "wheat_disease_golden"
REVIEWED = PROJECT / "wheat_pseudo_v5_reviewed"
RUST_REVIEW = PROJECT / "rust_review"       # 锈病终审
OUT = PROJECT / "wheat_v7_clean"
SRC_IMAGES = PROJECT / "wheat_disease_8class"

CLS_NAMES = ['Brown_Rust','Yellow_Rust','Black_Rust','Septoria',
             'Powdery_Mildew','Fusarium_HB','Healthy']

if OUT.exists():
    shutil.rmtree(OUT)
for split in ["train", "val"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

def find_image(stem):
    for s in ["train", "val"]:
        p = SRC_IMAGES / s / "images" / f"{stem}.jpg"
        if p.exists():
            return p
    for s in ["train", "val"]:
        p = GOLDEN / s / "images" / f"{stem}.jpg"
        if p.exists():
            return p
    return None

# 锈病终审标签索引
rust_labels = {}
if (RUST_REVIEW / "labels").exists():
    for lbl in (RUST_REVIEW / "labels").glob("*.txt"):
        rust_labels[lbl.stem] = lbl
print(f"锈病终审: {len(rust_labels)} 张")

seen = set()
golden_count = 0
reviewed_count = 0
rust_replaced = 0

# 1. 精标 (val保持原样)
for split in ["train", "val"]:
    img_dir = GOLDEN / split / "images"
    lbl_dir = GOLDEN / split / "labels"
    if not img_dir.exists():
        continue
    for img in img_dir.glob("*.jpg"):
        stem = img.stem
        if stem in seen:
            continue
        seen.add(stem)
        golden_count += 1

        # 锈病终审优先
        if stem in rust_labels:
            src_lbl = rust_labels[stem]
            rust_replaced += 1
        else:
            src_lbl = lbl_dir / f"{stem}.txt"

        if not src_lbl.exists():
            continue

        out_split = split
        shutil.copy2(img, OUT / out_split / "images" / img.name)
        shutil.copy2(src_lbl, OUT / out_split / "labels" / f"{stem}.txt")

# 2. 审核标签 (全部放train)
reviewed_lbl_dir = REVIEWED / "labels"
if reviewed_lbl_dir.exists():
    for lbl in reviewed_lbl_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in seen:
            continue
        seen.add(stem)
        reviewed_count += 1

        img = find_image(stem)
        if img is None:
            print(f"  MISSING: {stem}")
            continue

        # 锈病终审优先
        if stem in rust_labels:
            src_lbl = rust_labels[stem]
            rust_replaced += 1
        else:
            src_lbl = lbl

        shutil.copy2(img, OUT / "train" / "images" / img.name)
        shutil.copy2(src_lbl, OUT / "train" / "labels" / f"{stem}.txt")

print(f"\n精标: {golden_count} | 审核: {reviewed_count} | 合计: {golden_count + reviewed_count}")
print(f"锈病终审替换: {rust_replaced} 张")

# 统计
total = 0
cls_cnt = Counter()
for split in ["train", "val"]:
    ld = OUT / split / "labels"
    if not ld.exists():
        continue
    for f in ld.glob("*.txt"):
        total += 1
        for line in f.read_text().strip().splitlines():
            if line.strip():
                cls_cnt[int(line.split()[0])] += 1

print(f"\n总图: {total}")
for c in range(7):
    print(f"  [{c}] {CLS_NAMES[c]}: {cls_cnt.get(c,0)}")
print(f"\n→ {OUT}")
