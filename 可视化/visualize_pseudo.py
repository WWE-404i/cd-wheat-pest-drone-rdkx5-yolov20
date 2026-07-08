"""
可视化伪标注效果 — 画框 + 标签，方便人工审核
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
PSEUDO_DIR = PROJECT / "wheat_pseudo_v1"
OUT_DIR = PROJECT / "pseudo_review"
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Powdery Mildew", "Fusarium Head Blight", "Healthy Wheat",
]
COLORS = [
    "#E74C3C", "#F39C12", "#8E44AD", "#27AE60",
    "#2980B9", "#E67E22", "#95A5A6",
]


def draw_boxes(img_path, lbl_path, out_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # 线宽按图片大小缩放
    lw = max(4, int(min(w, h) / 200))
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, lw * 4))
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h

            color = COLORS[cls_id % len(COLORS)]
            # 画粗框 + 半透明填充
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            # 标签背景
            label = CLASS_NAMES[cls_id]
            bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
            draw.text((x1, y1 - lw - bbox[3] + bbox[1]), label, fill="white", font=font)

    img.save(out_path, quality=90)


def main():
    # 每类随机抽3张
    for cls_id in range(7):
        cls_name = CLASS_NAMES[cls_id]
        cls_dir = OUT_DIR / cls_name.replace(" ", "_")
        cls_dir.mkdir(exist_ok=True)

        # 找包含该类的伪标注图片
        candidates = []
        for lbl in (PSEUDO_DIR / "train" / "labels").glob("*.txt"):
            text = lbl.read_text().strip()
            if not text:
                continue
            for line in text.splitlines():
                if line.startswith(str(cls_id)):
                    candidates.append(lbl)
                    break

        if not candidates:
            print(f"{cls_name}: 无伪标注")
            continue

        random.shuffle(candidates)
        for lbl in random.sample(candidates, min(3, len(candidates))):
            img_path = PSEUDO_DIR / "train" / "images" / f"{lbl.stem}.jpg"
            if not img_path.exists():
                continue
            draw_boxes(img_path, lbl, cls_dir / f"{lbl.stem}_pseudo.jpg")
            print(f"  {cls_name}: {lbl.stem}")

    print(f"\n输出: {OUT_DIR}")
    print("每个子文件夹包含该类伪标注的可视化结果")


if __name__ == "__main__":
    main()
