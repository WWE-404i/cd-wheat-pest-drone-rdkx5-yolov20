"""
V6训练数据集构建：
- 精标534 + 审核192 + V5伪标注(6906,去掉Black_Rust) + V3伪标注(Black_Rust补充)
策略: V5主体(强类) + V3 Black_Rust(高Recall) + 人工审核纠正
"""
from pathlib import Path
import shutil

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN = PROJECT / "wheat_disease_golden"
V5_PSEUDO = PROJECT / "wheat_pseudo_v5"
V3_PSEUDO = PROJECT / "wheat_pseudo_v3"
REVIEWED = PROJECT / "wheat_pseudo_v5_reviewed"
OUT = PROJECT / "wheat_v6_combined"
SRC_IMAGES = PROJECT / "wheat_disease_8class"

OUT.mkdir(exist_ok=True)
for split in ["train", "val"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

stats = {"golden": 0, "reviewed": 0, "v5": 0, "v3_black": 0}

def find_image(stem):
    for s in ["train", "val"]:
        p = SRC_IMAGES / s / "images" / f"{stem}.jpg"
        if p.exists():
            return p
    return None

# ---------- 1. 精标集 ----------
for split in ["train", "val"]:
    img_dir = GOLDEN / split / "images"
    lbl_dir = GOLDEN / split / "labels"
    if not img_dir.exists():
        continue
    for img in img_dir.glob("*.jpg"):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        shutil.copy2(img, OUT / split / "images" / img.name)
        shutil.copy2(lbl, OUT / split / "labels" / lbl.name)
        stats["golden"] += 1
print(f"[1/4] 精标: {stats['golden']} 张")

# ---------- 2. 审核纠正的 ----------
reviewed_stems = set()
if REVIEWED.exists():
    out_lbl = REVIEWED / "labels"
    if out_lbl.exists():
        for lbl in out_lbl.glob("*.txt"):
            stem = lbl.stem
            img = find_image(stem)
            if img is None:
                continue
            # 放到train
            shutil.copy2(img, OUT / "train" / "images" / img.name)
            shutil.copy2(lbl, OUT / "train" / "labels" / lbl.name)
            reviewed_stems.add(stem)
            stats["reviewed"] += 1
print(f"[2/4] 审核纠正: {stats['reviewed']} 张")

# ---------- 3. V5 伪标注（主力，但去掉Black_Rust）----------
for split in ["train", "val"]:
    lbl_dir = V5_PSEUDO / split / "labels"
    if not lbl_dir.exists():
        continue
    for lbl in lbl_dir.glob("*.txt"):
        stem = lbl.stem
        if stem in reviewed_stems:
            continue  # 已被审核版替代

        # 读V5标签，去掉Black_Rust(2)
        lines = lbl.read_text().strip().splitlines()
        filtered = [l for l in lines if l.strip() and int(l.split()[0]) != 2]
        if not filtered:
            continue

        img = find_image(stem)
        if img is None:
            continue

        shutil.copy2(img, OUT / "train" / "images" / img.name)
        (OUT / "train" / "labels" / lbl.name).write_text("\n".join(filtered))
        stats["v5"] += 1
print(f"[3/4] V5伪标注(去掉Black_Rust): {stats['v5']} 张")

# ---------- 4. V3 伪标注 Black_Rust 补充 ----------
for split in ["train", "val"]:
    lbl_dir = V3_PSEUDO / split / "labels"
    if not lbl_dir.exists():
        continue
    for lbl in lbl_dir.glob("*.txt"):
        # 只取Black_Rust(2)
        lines = lbl.read_text().strip().splitlines()
        black_lines = [l for l in lines if l.strip() and int(l.split()[0]) == 2]
        if not black_lines:
            continue

        stem = lbl.stem
        img = find_image(stem)
        if img is None:
            continue

        out_img = OUT / "train" / "images" / img.name
        out_lbl = OUT / "train" / "labels" / f"{stem}.txt"

        if out_lbl.exists():
            # 该图已有label（来自精标/审核/V5），追加Black_Rust
            existing = out_lbl.read_text().strip().splitlines()
            combined = existing + black_lines
            out_lbl.write_text("\n".join(combined))
        else:
            shutil.copy2(img, out_img)
            out_lbl.write_text("\n".join(black_lines))
        stats["v3_black"] += 1
print(f"[4/4] V3 Black_Rust补充: {stats['v3_black']} 张")

# ---------- 统计 ----------
print(f"\n{'='*50}")
print("V6 数据集汇总")
print(f"{'='*50}")
total_imgs = 0
total_boxes = 0
per_class = {i: 0 for i in range(7)}
names = ["Brown_Rust","Yellow_Rust","Black_Rust","Septoria","Powdery_Mildew","Fusarium_HB","Healthy"]
for split in ["train", "val"]:
    lbl_dir = OUT / split / "labels"
    if not lbl_dir.exists():
        continue
    imgs = list(lbl_dir.glob("*.txt"))
    total_imgs += len(imgs)
    for lbl in imgs:
        for line in lbl.read_text().strip().splitlines():
            if line.strip():
                c = int(line.split()[0])
                if 0 <= c < 7:
                    per_class[c] += 1
                    total_boxes += 1
    print(f"  {split}: {len(imgs)} 张")

print(f"总图: {total_imgs}, 总框: {total_boxes}")
for c in range(7):
    print(f"  [{c}] {names[c]}: {per_class[c]}")
print(f"\n→ {OUT}")
