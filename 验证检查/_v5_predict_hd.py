"""V5预测演示 — 从手动标注数据集选高清图"""
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"
OUT = PROJECT / "v5_demo_hd"
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

# ===== 收集所有来源的标注图 =====
sources = {}

# 1. Golden set train
golden_train_img = PROJECT / "wheat_disease_golden" / "train" / "images"
golden_train_lbl = PROJECT / "wheat_disease_golden" / "train" / "labels"
if golden_train_img.exists():
    imgs = list(golden_train_img.glob("*.jpg")) + list(golden_train_img.glob("*.png"))
    for img in imgs:
        lbl = golden_train_lbl / (img.stem + ".txt")
        if lbl.exists():
            sources[img] = lbl

# 2. Golden set val
golden_val_img = PROJECT / "wheat_disease_golden" / "val" / "images"
golden_val_lbl = PROJECT / "wheat_disease_golden" / "val" / "labels"
if golden_val_img.exists():
    imgs = list(golden_val_img.glob("*.jpg")) + list(golden_val_img.glob("*.png"))
    for img in imgs:
        lbl = golden_val_lbl / (img.stem + ".txt")
        if lbl.exists():
            sources[img] = lbl

# 3. V9 seed val
v9_img = PROJECT / "wheat_v9_seed" / "val" / "images"
v9_lbl = PROJECT / "wheat_v9_seed" / "val" / "labels"
if v9_img.exists():
    imgs = list(v9_img.glob("*.jpg")) + list(v9_img.glob("*.png"))
    for img in imgs:
        lbl = v9_lbl / (img.stem + ".txt")
        if lbl.exists():
            sources[img] = lbl

# 4. V9 seed train (if exists)
v9t_img = PROJECT / "wheat_v9_seed" / "train" / "images"
v9t_lbl = PROJECT / "wheat_v9_seed" / "train" / "labels"
if v9t_img.exists():
    imgs = list(v9t_img.glob("*.jpg")) + list(v9t_img.glob("*.png"))
    for img in imgs:
        lbl = v9t_lbl / (img.stem + ".txt")
        if lbl.exists():
            sources[img] = lbl

print(f"总标注图: {len(sources)}")

# ===== 按类别分组，记录文件大小 =====
by_cls = defaultdict(list)  # {class_id: [(size, img_path), ...]}
for img, lbl in sources.items():
    # 读取标签获取类别
    classes = set()
    try:
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    c = int(line.split()[0])
                    classes.add(c)
    except:
        continue
    size = img.stat().st_size
    for c in classes:
        by_cls[c].append((size, img))

# ===== 每类选最大的4张 =====
samples = []
seen = set()
for c in range(7):
    items = sorted(by_cls[c], key=lambda x: x[0], reverse=True)  # 按文件大小降序
    count = 0
    for sz, img in items:
        if img not in seen:
            samples.append(img)
            seen.add(img)
            count += 1
            if count >= 4:
                break
    print(f"[{c}] {CLS_NAMES[c]}: {len(by_cls[c])}张标注图 → 选{count}张最大")

# 再随机补到30张
import random
remaining = [img for img in sources if img not in seen]
if remaining:
    extra = random.sample(remaining, min(max(0, 30 - len(samples)), len(remaining)))
    samples.extend(extra)
    print(f"额外补{len(extra)}张随机图")

print(f"\nV5模型: {MODEL}")
print(f"采样{len(samples)}张图预测...")
for i, s in enumerate(samples):
    kb = s.stat().st_size / 1024
    print(f"  {i+1}. [{kb:.0f}KB] {s.stem}")

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
img_dets = Counter()  # 多少张图有检出
for r in results:
    cls_list = [int(c) for c in r.boxes.cls.tolist()]
    if cls_list:
        img_dets["with_dets"] += 1
    else:
        img_dets["no_dets"] += 1
    for c in cls_list:
        all_dets[c] += 1

print(f"\n检测结果分布 ({img_dets['with_dets']}张有检出, {img_dets['no_dets']}张无检出):")
for c in range(7):
    print(f"  [{c}] {CLS_NAMES[c]}: {all_dets.get(c,0)}")
