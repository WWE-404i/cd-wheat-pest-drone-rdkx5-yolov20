"""
V8: 锈病人工审 + 非锈病V5伪标
策略: 锈病(0,1,2)=精标+审核终审，非锈病(3,4,5)=V5伪标保留
"""
from pathlib import Path
import shutil
from collections import Counter

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN = PROJECT / "wheat_disease_golden"
REVIEWED = PROJECT / "wheat_pseudo_v5_reviewed"
RUST_REVIEW = PROJECT / "rust_review"
V5_PSEUDO = PROJECT / "wheat_pseudo_v5"
OUT = PROJECT / "wheat_v8_hybrid"
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
        if p.exists():
            return p
    for s in ["train", "val"]:
        p = GOLDEN / s / "images" / f"{stem}.jpg"
        if p.exists():
            return p
    return None

def parse_label(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        if not line.strip(): continue
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes

def filter_nonrust(src_lbl_path, dst_lbl_path):
    """只保留非锈病类"""
    boxes = parse_label(src_lbl_path)
    filtered = [b for b in boxes if b[0] not in RUST_CLASSES]
    if filtered:
        lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in filtered]
        dst_lbl_path.write_text("\n".join(lines))
        return len(filtered) > 0
    return False

# 锈病终审标签索引
rust_labels = {}
if (RUST_REVIEW / "labels").exists():
    for lbl in (RUST_REVIEW / "labels").glob("*.txt"):
        rust_labels[lbl.stem] = lbl
print(f"锈病终审: {len(rust_labels)} 张")

seen = set()
golden_count = 0
reviewed_count = 0
v5_count = 0
rust_from_review = 0

# === 1. 精标 (完整保留, 锈病用终审版) ===
for split in ["train", "val"]:
    img_dir = GOLDEN / split / "images"
    lbl_dir = GOLDEN / split / "labels"
    if not img_dir.exists():
        continue
    for img in img_dir.glob("*.jpg"):
        stem = img.stem
        if stem in seen: continue
        seen.add(stem)
        golden_count += 1

        if stem in rust_labels:
            src_lbl = rust_labels[stem]
            rust_from_review += 1
        else:
            src_lbl = lbl_dir / f"{stem}.txt"

        if not src_lbl.exists(): continue
        shutil.copy2(img, OUT / split / "images" / img.name)
        shutil.copy2(src_lbl, OUT / split / "labels" / f"{stem}.txt")

# === 2. 审核标签 (完整保留, 锈病用终审版) ===
rev_lbl_dir = REVIEWED / "labels"
if rev_lbl_dir.exists():
    for lbl in rev_lbl_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in seen: continue
        seen.add(stem)
        reviewed_count += 1

        img = find_image(stem)
        if img is None: continue

        if stem in rust_labels:
            src_lbl = rust_labels[stem]
            rust_from_review += 1
        else:
            src_lbl = lbl

        shutil.copy2(img, OUT / "train" / "images" / img.name)
        shutil.copy2(src_lbl, OUT / "train" / "labels" / f"{stem}.txt")

# === 3. V5伪标: 只保留非锈病类(3,4,5,6) ===
v5_skipped_noimg = 0
v5_skipped_empty = 0
for split in ["train", "val"]:
    lbl_dir = V5_PSEUDO / split / "labels"
    if not lbl_dir.exists(): continue
    for lbl in lbl_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in seen: continue
        seen.add(stem)

        img = find_image(stem)
        if img is None:
            v5_skipped_noimg += 1
            continue

        # 过滤: 只保留非锈病类
        dst_lbl = OUT / "train" / "labels" / f"{stem}.txt"
        if filter_nonrust(lbl, dst_lbl):
            shutil.copy2(img, OUT / "train" / "images" / img.name)
            v5_count += 1
        else:
            v5_skipped_empty += 1

print(f"\n精标: {golden_count} | 审核: {reviewed_count} | V5非锈病: {v5_count}")
print(f"锈病终审替换: {rust_from_review} 张")
print(f"V5跳过(无图/无非锈病): {v5_skipped_noimg + v5_skipped_empty}")

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
