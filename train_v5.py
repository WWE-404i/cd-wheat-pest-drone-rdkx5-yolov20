"""
V5: 534精标 + 5806伪标注(V3) 合并训练
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN_DIR = PROJECT / "wheat_disease_golden"
PSEUDO_DIR = PROJECT / "wheat_pseudo_v3"
MERGED_DIR = PROJECT / "wheat_7class_v5"
V3_WEIGHTS = PROJECT / "runs" / "wheat_7class_v33" / "weights" / "best.pt"
SEED = 42


def merge():
    random.seed(SEED)
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for s in ["train", "val"]:
        (MERGED_DIR / s / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / s / "labels").mkdir(parents=True, exist_ok=True)

    # 收集精标
    items = []
    for split in ["train", "val"]:
        img_dir = GOLDEN_DIR / split / "images"
        lbl_dir = GOLDEN_DIR / split / "labels"
        if img_dir.exists():
            for img in img_dir.glob("*"):
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    items.append((img, lbl, "golden"))

    n_golden = len(items)

    # 收集伪标注
    pseudo_count = 0
    for split in ["train", "val"]:
        img_dir = PSEUDO_DIR / split / "images"
        lbl_dir = PSEUDO_DIR / split / "labels"
        if img_dir.exists():
            for img in img_dir.glob("*"):
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists() and lbl.stat().st_size > 0:
                    items.append((img, lbl, "pseudo"))
                    pseudo_count += 1

    print(f"Golden: {n_golden}, Pseudo: {pseudo_count}, Total: {len(items)}")

    # 按文件名去重
    seen = set()
    unique = []
    dup = 0
    for img, lbl, src in items:
        if img.name in seen:
            dup += 1
            continue
        seen.add(img.name)
        unique.append((img, lbl))
    if dup:
        print(f"去重: {dup} 张")

    random.shuffle(unique)
    n_val = max(50, int(len(unique) * 0.08))
    train_items = unique[n_val:]
    val_items = unique[:n_val]

    for img, lbl in train_items:
        shutil.copy2(img, MERGED_DIR / "train" / "images" / img.name)
        shutil.copy2(lbl, MERGED_DIR / "train" / "labels" / lbl.name if lbl else MERGED_DIR / "train" / "labels" / f"{img.stem}.txt")
    for img, lbl in val_items:
        shutil.copy2(img, MERGED_DIR / "val" / "images" / img.name)
        shutil.copy2(lbl, MERGED_DIR / "val" / "labels" / lbl.name if lbl else MERGED_DIR / "val" / "labels" / f"{img.stem}.txt")

    print(f"Train: {len(train_items)}, Val: {len(val_items)}")

    (MERGED_DIR / "data.yaml").write_text(f"""path: {MERGED_DIR}
train: train/images
val: val/images
nc: 7
names:
  0: Brown_Rust
  1: Yellow_Rust
  2: Black_Rust
  3: Septoria
  4: Powdery_Mildew
  5: Fusarium_Head_Blight
  6: Healthy_Wheat
""", encoding="utf-8")


def train():
    model = YOLO(str(V3_WEIGHTS))
    model.train(
        data=str(MERGED_DIR / "data.yaml"),
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,

        # 数据量够了，适度增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        perspective=0.0005, flipud=0.3, fliplr=0.5,
        mosaic=1.0, mixup=0.1, close_mosaic=30,

        val=True, save=True, save_period=30, patience=25,
        project=str(PROJECT / "runs"), name="wheat_7class_v5",
    )


if __name__ == "__main__":
    merge()
    train()
