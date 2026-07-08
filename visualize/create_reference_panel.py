"""
将参考裁剪样本拼成4行对照图，加标签
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
CROP_DIR = PROJECT / "reference_crops"
OUT_PATH = PROJECT / "reference_panel.png"

CLASS_NAMES = ["Brown_Rust", "Yellow_Rust", "Black_Rust", "Septoria"]
CLASS_LABELS = [
    "0 Brown_Rust 叶锈 - 圆形散落",
    "1 Yellow_Rust 条锈 - 成条排列",
    "2 Black_Rust 秆锈 - 大斑撕裂",
    "3 Septoria 叶枯 - 褐斑黑点",
]
COLORS = ["#FF6B35", "#FFD23F", "#CC0000", "#2E7D32"]

COLS = 10
ROWS = 2
CROP_SIZE = 120
LABEL_H = 30
GAP = 4
FONT_SIZE = 14

# 加载字体
try:
    font = ImageFont.truetype("msyh.ttc", FONT_SIZE)
except Exception:
    font = ImageFont.load_default()

panel_w = COLS * CROP_SIZE + (COLS - 1) * GAP + 20
panel_h = len(CLASS_NAMES) * (ROWS * CROP_SIZE + (ROWS - 1) * GAP + LABEL_H + GAP) + 20
panel = Image.new("RGB", (panel_w, panel_h), "#1A1A1A")
draw = ImageDraw.Draw(panel)

y_offset = 10
for cid, name in enumerate(CLASS_NAMES):
    cls_dir = CROP_DIR / name
    crops = sorted(cls_dir.glob("*.jpg"))
    if not crops:
        continue

    # 类别标签
    draw.text((10, y_offset), CLASS_LABELS[cid], fill=COLORS[cid], font=font)
    y_offset += LABEL_H + GAP

    # 拼两行
    for row in range(ROWS):
        x_offset = 10
        for col in range(COLS):
            idx = row * COLS + col
            if idx < len(crops):
                try:
                    img = Image.open(crops[idx])
                    panel.paste(img, (x_offset, y_offset))
                except Exception:
                    pass
            x_offset += CROP_SIZE + GAP
        y_offset += CROP_SIZE + GAP

    y_offset += GAP

panel.save(OUT_PATH)
print(f"参考对照图: {OUT_PATH} ({panel.size})")
for cid, name in enumerate(CLASS_NAMES):
    n = len(list((CROP_DIR / name).glob("*.jpg")))
    print(f"  {CLASS_LABELS[cid]}: {n} 样本")
