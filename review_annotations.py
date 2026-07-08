"""
标注可视化复检 — 每类随机抽样N张，画bbox输出到 review/ 目录
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import colorsys

# ========== 配置 ==========
DATASET_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_yolo_train")
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\review")
SAMPLES_PER_CLASS = 10

CLASS_NAMES = [
    "Brown Rust",
    "Yellow Rust",
    "Black Rust",
    "Septoria",
    "Mildew",
    "Fusarium Head Blight",
    "Healthy Wheat",
]

# 每类一个颜色
COLORS = []
for i in range(len(CLASS_NAMES)):
    hue = i / len(CLASS_NAMES)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    COLORS.append((int(r*255), int(g*255), int(b*255)))


def draw_boxes(image_path, label_path, output_path):
    """在图片上画 YOLO 格式的 bbox"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if not label_path.exists():
        # 无标注 — 画一个红叉标记
        draw.line([(0, 0), (w, h)], fill=(255, 0, 0), width=3)
        draw.line([(w, 0), (0, h)], fill=(255, 0, 0), width=3)
        draw.text((10, 10), "NO LABEL", fill=(255, 0, 0))
        img.save(output_path, quality=90)
        return 0

    # 读 YOLO 标注
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, bw, bh = float(parts[1]), float(parts[3]), float(parts[4])

            x1 = int((cx - bw / 2) * w)
            y1 = int((float(parts[2]) - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((float(parts[2]) + bh / 2) * h)

            boxes.append((cls_id, x1, y1, x2, y2))

    if not boxes:
        draw.text((10, 10), "EMPTY", fill=(255, 165, 0))
        img.save(output_path, quality=90)
        return 0

    # 画框
    for cls_id, x1, y1, x2, y2 in boxes:
        color = COLORS[cls_id % len(COLORS)]
        name = CLASS_NAMES[cls_id]
        # 框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # 标签
        label = f"{cls_id}:{name[:6]}"
        tw = len(label) * 7
        draw.rectangle([x1, y1-16, x1+tw, y1], fill=color)
        draw.text((x1+2, y1-15), label, fill=(255, 255, 255))

    img.save(output_path, quality=90)
    return len(boxes)


def main():
    print("=" * 60)
    print("标注可视化复检")
    print(f"每类抽样: {SAMPLES_PER_CLASS} 张")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 查找图片来源 (优先 train, 不够用 val 补)
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        if not img_dir.exists():
            continue

        # 收集每一类的图片
        class_images = {name: [] for name in CLASS_NAMES}
        for img_path in img_dir.glob("*.jpg"):
            for cls_name in CLASS_NAMES:
                if img_path.name.startswith(f"{cls_name}_"):
                    class_images[cls_name].append((img_path, split))
                    break

        # 抽样 + 画图
        total_boxes = 0
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            imgs = class_images[cls_name]
            if not imgs:
                continue

            sample = random.sample(imgs, min(SAMPLES_PER_CLASS, len(imgs)))

            for j, (img_path, s) in enumerate(sample):
                lbl_path = DATASET_DIR / s / "labels" / f"{img_path.stem}.txt"
                out_name = f"{cls_id}_{cls_name}_{j+1}.jpg"
                out_path = OUTPUT_DIR / out_name

                n = draw_boxes(img_path, lbl_path, out_path)
                total_boxes += n

        print(f"\n生成 {total_boxes} 个框的可视化图片")
        print(f"查看: {OUTPUT_DIR}")
        print(f"\n类别颜色对照:")
        for i, name in enumerate(CLASS_NAMES):
            r, g, b = COLORS[i]
            print(f"  [{i}] {name} — rgb({r},{g},{b})")


if __name__ == "__main__":
    main()
