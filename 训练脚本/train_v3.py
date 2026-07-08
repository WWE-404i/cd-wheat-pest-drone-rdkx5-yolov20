"""
V3: 383精标 + 3075伪标注(V2) → 合并训练
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MANUAL_DIR = PROJECT / "wheat_7class_manual"
PSEUDO_DIR = PROJECT / "wheat_pseudo_v2"
MERGED_DIR = PROJECT / "wheat_7class_v3"
SEED = 42


def merge():
    random.seed(SEED)
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for s in ["train", "val"]:
        (MERGED_DIR / s / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / s / "labels").mkdir(parents=True, exist_ok=True)

    items = []
    for s in ["train", "val"]:
        for img in (MANUAL_DIR / s / "images").glob("*"):
            lbl = MANUAL_DIR / s / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                items.append((img, lbl))
    for s in ["train", "val"]:
        for img in (PSEUDO_DIR / s / "images").glob("*"):
            lbl = PSEUDO_DIR / s / "labels" / f"{img.stem}.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                items.append((img, lbl))

    print(f"Total: {len(items)} images ({len(items) - 383} pseudo)")

    random.shuffle(items)
    n_val = max(30, int(len(items) * 0.08))
    for img, lbl in items[n_val:]:
        shutil.copy2(img, MERGED_DIR / "train" / "images" / img.name)
        shutil.copy2(lbl, MERGED_DIR / "train" / "labels" / lbl.name)
    for img, lbl in items[:n_val]:
        shutil.copy2(img, MERGED_DIR / "val" / "images" / img.name)
        shutil.copy2(lbl, MERGED_DIR / "val" / "labels" / lbl.name)

    print(f"Train: {len(items) - n_val}, Val: {n_val}")

    (MERGED_DIR / "data.yaml").write_text(f"""path: {MERGED_DIR}
train: train/images
val: val/images
nc: 7
names:
  0: Brown Rust
  1: Yellow Rust
  2: Black Rust
  3: Septoria
  4: Powdery Mildew
  5: Fusarium Head Blight
  6: Healthy Wheat
""", encoding="utf-8")


def train():
    model = YOLO(PROJECT / "yolo26n.pt")
    model.train(
        data=str(MERGED_DIR / "data.yaml"),
        epochs=200, imgsz=640, batch=16, device=0, workers=4,
        project=str(PROJECT / "runs"), name="wheat_7class_v3",
        optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.9,
        weight_decay=0.0005, warmup_epochs=3,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        perspective=0.0005, flipud=0.3, fliplr=0.5,
        mosaic=1.0, mixup=0.1, close_mosaic=30,
        val=True, save=True, save_period=30, patience=25,
    )


if __name__ == '__main__':
    merge()
    train()
