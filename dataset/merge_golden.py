"""
合并三个精标数据集 → 统一 YOLO 格式，train/val 切分
来源: golden_set_7class(158) + to_label(225) + to_label_v2(243)
注意: to_label 标签文件名格式为 {类名}_{原始名}.txt，需匹配类文件夹中的原始图片
"""
from pathlib import Path
import shutil
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
OUT_DIR = PROJECT / "wheat_disease_golden"

VAL_RATIO = 0.15
SEED = 42

CLASS_NAMES = [
    "Brown_Rust", "Yellow_Rust", "Black_Rust", "Septoria",
    "Powdery_Mildew", "Fusarium_Head_Blight", "Healthy_Wheat",
]

# to_label 类文件夹名映射（可能带空格或下划线）
CLASS_DIR_NAMES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Mildew", "Fusarium Head Blight", "Healthy Wheat",
]


def strip_class_prefix(label_stem):
    """尝试去掉标签文件名的类名前缀，返回可能的原始图片名列表"""
    candidates = [label_stem]  # 先尝试直接匹配
    for cls_name in CLASS_DIR_NAMES:
        # 标签格式: "Brown Rust_Brown Rust_xxx" → 原图: "Brown Rust_xxx"
        prefix = cls_name + "_"
        if label_stem.startswith(prefix):
            candidates.insert(0, label_stem[len(prefix):])
    return candidates


def collect_pairs():
    """收集所有 (img_path, lbl_path) 对"""
    pairs = []

    # 1. golden_set_7class — 标准 YOLO 格式
    gs = PROJECT / "golden_set_7class"
    for split in ["train", "val"]:
        img_dir = gs / split / "images"
        lbl_dir = gs / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))
            else:
                pairs.append((img, None))

    # 2. to_label — images/ + 类子文件夹，匹配 labels/
    tl = PROJECT / "to_label"
    tl_lbls = tl / "labels"
    # 构建类文件夹图片索引
    cls_images = {}  # stem → path
    for cls_dir in tl.iterdir():
        if not cls_dir.is_dir() or cls_dir.name in ("images", "labels"):
            continue
        for img in cls_dir.glob("*.jpg"):
            cls_images[img.stem] = img
    # 也加入 images/ 的
    tl_imgs_dir = tl / "images"
    if tl_imgs_dir.exists():
        for img in tl_imgs_dir.glob("*.jpg"):
            cls_images[img.stem] = img

    if tl_lbls.exists():
        for lbl in tl_lbls.glob("*.txt"):
            candidates = strip_class_prefix(lbl.stem)
            found = None
            for c in candidates:
                if c in cls_images:
                    found = cls_images[c]
                    break
            if found:
                pairs.append((found, lbl))

    # 3. to_label_v2 — images/ + labels/
    v2 = PROJECT / "to_label_v2"
    v2_imgs = v2 / "images"
    v2_lbls = v2 / "labels"
    if v2_imgs.exists():
        for img in v2_imgs.glob("*.jpg"):
            lbl = v2_lbls / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))

    return pairs


def main():
    pairs = collect_pairs()
    print(f"收集到 {len(pairs)} 个图像-标签对")

    # 按文件名去重（保留先遇到的）
    seen_names = set()
    seen_stems = set()
    unique_pairs = []
    dup_count = 0
    for img, lbl in pairs:
        if img.stem in seen_stems:
            dup_count += 1
            continue
        seen_stems.add(img.stem)
        seen_names.add(img.name)
        unique_pairs.append((img, lbl))

    if dup_count:
        print(f"去重: 删除 {dup_count} 张重复")
    print(f"去重后: {len(unique_pairs)} 张")

    # 统计
    total_boxes = 0
    cls_box_counts = {i: 0 for i in range(7)}
    empty_count = 0
    for _, lbl in unique_pairs:
        content = lbl.read_text().strip() if lbl else ""
        if not content:
            empty_count += 1
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if 0 <= cls_id <= 6:
                    cls_box_counts[cls_id] += 1
                    total_boxes += 1

    print(f"\n总框数: {total_boxes}")
    print(f"空标签(Healthy): {empty_count} 张")
    print(f"平均框/图: {total_boxes / len(unique_pairs):.2f}")
    print("各类框数:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}: {cls_box_counts[i]}")

    # 按第一框类别分层切分 train/val
    random.seed(SEED)
    class_images = {i: [] for i in range(7)}
    for img, lbl in unique_pairs:
        if lbl and lbl.exists():
            content = lbl.read_text().strip()
            if content:
                first_cls = int(content.splitlines()[0].split()[0])
                class_images[first_cls].append((img, lbl))
                continue
        class_images[6].append((img, lbl))

    train_pairs, val_pairs = [], []
    for cls_id in range(7):
        items = class_images[cls_id]
        random.shuffle(items)
        n_val = max(1, int(len(items) * VAL_RATIO))
        val_pairs.extend(items[:n_val])
        train_pairs.extend(items[n_val:])

    print(f"\nTrain: {len(train_pairs)}, Val: {len(val_pairs)}")

    # 输出
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split, pairs_list in [("train", train_pairs), ("val", val_pairs)]:
        (OUT_DIR / split / "images").mkdir(parents=True)
        (OUT_DIR / split / "labels").mkdir(parents=True)
        for img, lbl in pairs_list:
            dst_img = OUT_DIR / split / "images" / img.name
            # 处理重名冲突
            if dst_img.exists():
                dst_img = OUT_DIR / split / "images" / f"{img.stem}_{hash(img)}_{img.suffix}"
            shutil.copy2(img, dst_img)
            if lbl and lbl.exists():
                dst_lbl = OUT_DIR / split / "labels" / f"{dst_img.stem}.txt"
                shutil.copy2(lbl, dst_lbl)
            else:
                (OUT_DIR / split / "labels" / f"{dst_img.stem}.txt").touch()

    # data.yaml
    yaml_content = f"""# Wheat Disease Golden Dataset — 7 classes
path: {OUT_DIR}
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
"""
    (OUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")

    final_train = len(list((OUT_DIR / "train" / "images").glob("*.jpg")))
    final_val = len(list((OUT_DIR / "val" / "images").glob("*.jpg")))
    print(f"\n输出: {OUT_DIR}")
    print(f"  train: {final_train} 张")
    print(f"  val: {final_val} 张")
    print(f"  合计: {final_train + final_val} 张")
    print("完成!")


if __name__ == "__main__":
    main()
