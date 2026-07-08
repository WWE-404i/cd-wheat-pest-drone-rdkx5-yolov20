"""
V6b 数据集: V5伪标注基础 + 审核192替换，不混V3
"""
from pathlib import Path
import shutil

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN = PROJECT / "wheat_disease_golden"
V5_PSEUDO = PROJECT / "wheat_pseudo_v5"
REVIEWED = PROJECT / "wheat_pseudo_v5_reviewed"
OUT = PROJECT / "wheat_v6b_combined"
SRC_IMAGES = PROJECT / "wheat_disease_8class"

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
    return None

# 审核过的 stem
reviewed_stems = set()
if (REVIEWED / "labels").exists():
    for lbl in (REVIEWED / "labels").glob("*.txt"):
        reviewed_stems.add(lbl.stem)
print(f"已审核: {len(reviewed_stems)} 张")

golden_stems = set()
# 1. 精标 (val保持)
for split in ["train", "val"]:
    img_dir = GOLDEN / split / "images"
    lbl_dir = GOLDEN / split / "labels"
    if not img_dir.exists():
        continue
    for img in img_dir.glob("*.jpg"):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        golden_stems.add(img.stem)
        shutil.copy2(img, OUT / split / "images" / img.name)
        shutil.copy2(lbl, OUT / split / "labels" / lbl.name)
print(f"精标: {len(golden_stems)} 张")

# 2. V5伪标注 + 审核替换
v5_count = 0
reviewed_used = 0
for split in ["train", "val"]:
    lbl_dir = V5_PSEUDO / split / "labels"
    if not lbl_dir.exists():
        continue
    for lbl in lbl_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in golden_stems:
            continue

        # 审核版优先
        if stem in reviewed_stems:
            src_lbl = REVIEWED / "labels" / f"{stem}.txt"
            reviewed_used += 1
        else:
            src_lbl = lbl

        img = find_image(stem)
        if img is None:
            continue

        shutil.copy2(img, OUT / "train" / "images" / img.name)
        shutil.copy2(src_lbl, OUT / "train" / "labels" / f"{stem}.txt")
        v5_count += 1

print(f"V5伪标注: {v5_count} 张 (其中审核替换: {reviewed_used})")

# 统计
total = 0
import collections
cls_cnt = collections.Counter()
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
names = ['Brown_Rust','Yellow_Rust','Black_Rust','Septoria','Powdery_Mildew','Fusarium_HB','Healthy']
for c in range(7):
    print(f"  [{c}] {names[c]}: {cls_cnt.get(c,0)}")
print(f"\n→ {OUT}")
