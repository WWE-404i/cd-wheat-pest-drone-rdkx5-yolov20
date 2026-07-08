"""生成四类病变对比参考图"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REF = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\reference_crops")
CLASSES = ["Brown_Rust", "Yellow_Rust", "Black_Rust", "Septoria"]
CLASS_NAMES_CN = ["0:叶锈(散落)", "1:条锈(成行)", "2:秆锈(大斑)", "3:叶枯(黑点)"]

CROP_SIZE = 120
COLS = 10  # 每行10个
ROWS = 2   # 每类2行
GAP = 4
HEADER_H = 28

# 找字体
font = None
for f in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc",
          "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/arial.ttf"]:
    p = Path(f)
    if p.exists():
        try:
            font = ImageFont.truetype(str(p), 16)
        except:
            pass
        if font:
            break
if font is None:
    font = ImageFont.load_default()

panel_w = COLS * (CROP_SIZE + GAP) + GAP
panel_h = HEADER_H + ROWS * (CROP_SIZE + GAP) + GAP

# 每个 class 一个 panel
panels = []
for cls_name, cn_name in zip(CLASSES, CLASS_NAMES_CN):
    panel = Image.new("RGB", (panel_w, panel_h), (40, 40, 40))
    draw = ImageDraw.Draw(panel)
    draw.text((GAP, 4), cn_name, fill=(255, 255, 255), font=font)

    crop_dir = REF / cls_name
    crops = sorted(crop_dir.glob("*.jpg"))[:COLS * ROWS]

    for idx, crop_path in enumerate(crops):
        row = idx // COLS
        col = idx % COLS
        x = GAP + col * (CROP_SIZE + GAP)
        y = HEADER_H + row * (CROP_SIZE + GAP)
        try:
            img = Image.open(crop_path)
            # 画边框
            panel.paste(img, (x, y))
            draw.rectangle([x, y, x + CROP_SIZE, y + CROP_SIZE],
                           outline=(100, 100, 100), width=1)
        except Exception:
            pass

    # 如果不满，填充空位
    for idx in range(len(crops), COLS * ROWS):
        row = idx // COLS
        col = idx % COLS
        x = GAP + col * (CROP_SIZE + GAP)
        y = HEADER_H + row * (CROP_SIZE + GAP)
        draw.rectangle([x, y, x + CROP_SIZE, y + CROP_SIZE],
                       fill=(30, 30, 30), outline=(60, 60, 60))

    panels.append(panel)

# 纵向拼接
total_h = panel_h * 4
total_w = panel_w
ref_img = Image.new("RGB", (total_w, total_h), (40, 40, 40))

for i, panel in enumerate(panels):
    ref_img.paste(panel, (0, i * panel_h))

out_path = REF / "comparison_reference.jpg"
ref_img.save(out_path, quality=90)
print(f"参考图: {out_path} ({total_w}x{total_h})")
