"""
从精标集提取四类病变(Brown/Yellow/Black/Septoria)的裁剪样本做参考
"""
from pathlib import Path
from PIL import Image

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN = PROJECT / "wheat_disease_golden"
OUT = PROJECT / "reference_crops"
CLASS_NAMES = ["Brown_Rust", "Yellow_Rust", "Black_Rust", "Septoria"]
CLASS_IDS = [0, 1, 2, 3]

OUT.mkdir(exist_ok=True)

for cid in CLASS_IDS:
    (OUT / CLASS_NAMES[cid]).mkdir(exist_ok=True)

counts = {c: 0 for c in CLASS_IDS}

for split in ["train", "val"]:
    img_dir = GOLDEN / split / "images"
    lbl_dir = GOLDEN / split / "labels"
    if not img_dir.exists():
        continue
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        img_path = img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            continue
        pil = Image.open(img_path)
        w, h = pil.size
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if cid not in CLASS_IDS:
                continue
            if counts[cid] >= 20:  # 每类最多20个
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil.crop((x1, y1, x2, y2))
            crop = crop.resize((120, 120), Image.LANCZOS)
            crop.save(OUT / CLASS_NAMES[cid] / f"{lbl_path.stem}_{counts[cid]:03d}.jpg")
            counts[cid] += 1

print("提取完成:")
for cid in CLASS_IDS:
    print(f"  {CLASS_NAMES[cid]}: {counts[cid]} 个")
