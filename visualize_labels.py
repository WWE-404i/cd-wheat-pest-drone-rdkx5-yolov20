"""
随机抽取 GD 标注图片，可视化检测框，用于人工质检。
解决 cv2 中文路径问题：读写全部用 imdecode/imencode。
"""
import cv2
import numpy as np
from pathlib import Path
import random
import os

# 输出到不含中文的路径
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\label_inspection")
PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
IMAGES_DIR = PROJECT / "wheat_yolo_train" / "train" / "images"
LABELS_DIR = PROJECT / "wheat_yolo_train" / "train" / "labels"
N_SAMPLES = 20

CLASSES = ["Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
           "Mildew", "Fusarium Head Blight", "Healthy Wheat"]
COLORS = [
    (0, 0, 255),      # Red - Brown Rust
    (0, 255, 255),    # Yellow - Yellow Rust
    (255, 255, 255),  # White - Black Rust
    (255, 0, 0),      # Blue - Septoria
    (255, 0, 255),    # Purple - Mildew
    (0, 128, 255),    # Orange - Fusarium HB
    (0, 255, 0),      # Green - Healthy Wheat
]

os.makedirs(str(OUTPUT_DIR), exist_ok=True)


def imread_cn(path):
    """cv2.imread with Chinese path support"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_cn(path, img):
    """cv2.imwrite with Chinese path support"""
    _, buf = cv2.imencode(Path(path).suffix, img)
    with open(path, 'wb') as f:
        f.write(buf)


def draw_boxes(img_path, label_path, out_path):
    img = imread_cn(str(img_path))
    if img is None:
        print(f"  SKIP: cannot decode {img_path.name}")
        return False

    h, w = img.shape[:2]

    if label_path.exists():
        with open(str(label_path)) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = COLORS[cls_id]
            name = CLASSES[cls_id]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, name, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        cv2.putText(img, "NO LABEL", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # watermark: filename
    cv2.putText(img, img_path.name, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    imwrite_cn(str(out_path), img)
    return True


def main():
    label_files = list(LABELS_DIR.glob("*.txt"))
    print(f"Total train labels: {len(label_files)}")

    # stratified sampling by class
    by_class = {}
    for lf in label_files:
        with open(str(lf)) as f:
            first = f.readline().strip()
        cid = int(first.split()[0]) if first else -1
        by_class.setdefault(cid, []).append(lf)

    sampled = []
    for cid, files in by_class.items():
        n = max(1, N_SAMPLES * len(files) // len(label_files))
        sampled.extend(random.sample(files, min(n, len(files))))

    if len(sampled) < N_SAMPLES:
        rest = [lf for lf in label_files if lf not in sampled]
        sampled.extend(random.sample(rest, min(N_SAMPLES - len(sampled), len(rest))))

    random.shuffle(sampled)
    sampled = sampled[:N_SAMPLES]

    print(f"Sampled {len(sampled)}, drawing boxes...\n")

    ok = 0
    for i, lf in enumerate(sampled):
        stem = lf.stem
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            c = IMAGES_DIR / (stem + ext)
            if c.exists():
                img = c
                break
        if img is None:
            print(f"  [{i+1}/{len(sampled)}] MISSING IMAGE: {stem}")
            continue

        out_name = f"{i+1:02d}_{lf.stem}.jpg"
        if draw_boxes(img, lf, OUTPUT_DIR / out_name):
            ok += 1
            # class stats
            with open(str(lf)) as f:
                boxes = [l.strip() for l in f if l.strip()]
            cc = {}
            for b in boxes:
                cn = CLASSES[int(b.split()[0])]
                cc[cn] = cc.get(cn, 0) + 1
            stats = ", ".join(f"{k}:{v}" for k, v in cc.items())
            print(f"  [{i+1}/{len(sampled)}] {out_name}  [{stats}]")

    print(f"\nDone! {ok}/{len(sampled)} images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
