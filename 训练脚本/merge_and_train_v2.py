"""
合并: wheat_7class_manual (383精标) + wheat_pseudo_v1 (1156伪标注)
训练 V2
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MANUAL_DIR = PROJECT / "wheat_7class_manual"
PSEUDO_DIR = PROJECT / "wheat_pseudo_v1"
MERGED_DIR = PROJECT / "wheat_7class_v2"

SEED = 42
VAL_RATIO = 0.1
random.seed(SEED)


def merge():
    """合并精标+伪标注，重新切分 train/val"""
    print("=" * 60)
    print("合并数据集: 精标 + 伪标注")
    print("=" * 60)

    # 清空
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    for split in ["train", "val"]:
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    all_items = []  # (img_path, lbl_path)

    # 1. 精标数据 (train中用)
    for img in (MANUAL_DIR / "train" / "images").glob("*"):
        lbl = MANUAL_DIR / "train" / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            all_items.append((img, lbl))

    # 也加入精标 val
    for img in (MANUAL_DIR / "val" / "images").glob("*"):
        lbl = MANUAL_DIR / "val" / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            all_items.append((img, lbl))

    print(f"精标: {len(all_items)} 张")

    # 2. 伪标注数据
    pseudo_count = 0
    for img in (PSEUDO_DIR / "train" / "images").glob("*"):
        lbl = PSEUDO_DIR / "train" / "labels" / f"{img.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:  # 有框的才加
            all_items.append((img, lbl))
            pseudo_count += 1

    for img in (PSEUDO_DIR / "val" / "images").glob("*"):
        lbl = PSEUDO_DIR / "val" / "labels" / f"{img.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:
            all_items.append((img, lbl))
            pseudo_count += 1

    print(f"伪标注(有框): {pseudo_count} 张")
    print(f"合并总计: {len(all_items)} 张")

    # 3. 重新切分 train/val
    random.shuffle(all_items)
    n_val = max(30, int(len(all_items) * VAL_RATIO))
    train_items = all_items[n_val:]
    val_items = all_items[:n_val]

    for img, lbl in train_items:
        shutil.copy2(img, MERGED_DIR / "train" / "images" / img.name)
        shutil.copy2(lbl, MERGED_DIR / "train" / "labels" / lbl.name)

    for img, lbl in val_items:
        shutil.copy2(img, MERGED_DIR / "val" / "images" / img.name)
        shutil.copy2(lbl, MERGED_DIR / "val" / "labels" / lbl.name)

    print(f"\nTrain: {len(train_items)}, Val: {len(val_items)}")

    # 统计各类框数
    cls_counts = {i: 0 for i in range(7)}
    for split in ["train", "val"]:
        for lbl in (MERGED_DIR / split / "labels").glob("*.txt"):
            for line in lbl.read_text().strip().splitlines():
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1

    CLASS_NAMES = ["Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
                   "Powdery Mildew", "Fusarium Head Blight", "Healthy Wheat"]
    print("\n各类框数:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}: {cls_counts[i]}")
    print(f"  总框: {sum(cls_counts.values())}")

    # 写 data.yaml
    yaml = f"""path: {MERGED_DIR}
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
"""
    (MERGED_DIR / "data.yaml").write_text(yaml, encoding="utf-8")


def train():
    """训练 V2"""
    print(f"\n{'='*60}")
    print("训练 V2")
    print("=" * 60)

    model = YOLO(PROJECT / "yolo26n.pt")
    results = model.train(
        data=str(MERGED_DIR / "data.yaml"),
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(PROJECT / "runs"),
        name="wheat_7class_v2",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=30,
        val=True,
        save=True,
        save_period=50,
        patience=50,
    )


if __name__ == "__main__":
    merge()
    train()
