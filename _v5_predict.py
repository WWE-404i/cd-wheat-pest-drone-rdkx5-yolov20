"""V5预测演示 — 在采样图上可视化"""
from ultralytics import YOLO
from pathlib import Path
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"
OUT = PROJECT / "v5_demo"
OUT.mkdir(exist_ok=True)

CLS_NAMES = ['Brown_Rust','Yellow_Rust','Black_Rust','Septoria',
             'Powdery_Mildew','Fusarium_HB','Healthy']
COLORS = {
    0: (0,0,255),    # 红 - Brown_Rust
    1: (0,255,255),  # 黄 - Yellow_Rust
    2: (255,0,0),    # 蓝 - Black_Rust
    3: (0,255,0),    # 绿 - Septoria
    4: (255,0,255),  # 紫 - Powdery_Mildew
    5: (255,255,0),  # 青 - Fusarium_HB
    6: (128,128,128),# 灰 - Healthy
}

# 找图：每种类别各取几张
SRC = PROJECT / "wheat_disease_8class" / "train" / "images"
all_imgs = list(SRC.glob("*.jpg"))

# 按文件名中的类名分组
by_cls = {c: [] for c in range(7)}
for img in all_imgs:
    stem = img.stem.lower()
    for i, name in enumerate(CLS_NAMES):
        if name.lower().replace('_',' ') in stem or name.lower() in stem:
            by_cls[i].append(img)
            if len(by_cls[i]) >= 5: break

# 采20张
samples = []
for c in range(7):
    if by_cls[c]:
        samples.extend(random.sample(by_cls[c], min(3, len(by_cls[c]))))
# 再随机补一些
remaining = random.sample(all_imgs, max(0, 25 - len(samples)))
samples.extend(remaining)

print(f"V5模型: {MODEL}")
print(f"采样{len(samples)}张图预测...")

model = YOLO(str(MODEL))
results = model.predict(
    source=[str(s) for s in samples],
    imgsz=640,
    conf=0.25,
    iou=0.7,
    save=True,
    project=str(OUT),
    name="",
    exist_ok=True,
    show_labels=True,
    show_conf=True,
)

print(f"\n预测结果: {OUT}/")
# 统计各类别检测数
from collections import Counter
all_dets = Counter()
for r in results:
    for c in r.boxes.cls.tolist():
        all_dets[int(c)] += 1
print("检测结果分布:")
for c in range(7):
    print(f"  [{c}] {CLS_NAMES[c]}: {all_dets.get(c,0)}")
