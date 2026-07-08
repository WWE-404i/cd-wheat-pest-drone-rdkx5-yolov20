"""
golden_set 标签转换：统一 class ID → 7 类系统
以文件名中的文件夹类名为准，忽略旧标签中的 class ID
输出: golden_set_7class/
"""
import shutil
import random
from pathlib import Path
from collections import Counter

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN_SRC = PROJECT / "golden_set"
GOLDEN_OUT = PROJECT / "golden_set_7class"

# ============================================
# 新 7 类映射（以文件夹名为准）
# ============================================
CLASS_MAP = {
    "Brown Rust": 0,
    "Yellow Rust": 1,
    "Black Rust": 2,
    "Septoria": 3,
    "Mildew": 4,       # → Powdery Mildew
    "Fusarium Head Blight": 5,
    "Healthy Wheat": 6,
}

CLASS_NAMES = [None] * 7
for name, cid in CLASS_MAP.items():
    CLASS_NAMES[cid] = name
# Mildew → Powdery Mildew 改名
CLASS_NAMES[4] = "Powdery Mildew"

SEED = 42
VAL_RATIO = 0.15

random.seed(SEED)


def extract_class_from_filename(filename_stem):
    """从文件名提取类别，如 'Brown Rust_Brown Rust_101' → 'Brown Rust'"""
    for cls_name in CLASS_MAP:
        if filename_stem.startswith(cls_name + "_"):
            return cls_name
    return None


def main():
    print("=" * 60)
    print("golden_set → 7 类标签转换")
    print("=" * 60)
    print(f"源: {GOLDEN_SRC}")
    print(f"目标: {GOLDEN_OUT}")
    print(f"类别:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}")

    # 清空输出
    if GOLDEN_OUT.exists():
        shutil.rmtree(GOLDEN_OUT)

    for split in ["train", "val"]:
        (GOLDEN_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (GOLDEN_OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    # 收集所有标注
    label_files = list((GOLDEN_SRC / "labels").glob("*.txt"))
    print(f"\n标注文件: {len(label_files)} 个")

    # 按类别分组，用于分层采样 split
    by_class = {name: [] for name in CLASS_MAP}
    unknown = []
    fixed_count = 0
    skipped = 0

    for lbl_path in label_files:
        stem = lbl_path.stem
        cls_name = extract_class_from_filename(stem)

        if cls_name is None:
            unknown.append(stem)
            continue

        new_cls_id = CLASS_MAP[cls_name]

        # 读旧标签
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # 保留 bbox 坐标，替换 class ID
            cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
            new_lines.append(f"{new_cls_id} {cx} {cy} {w} {h}")

        if not new_lines and cls_name != "Healthy Wheat":
            # 非健康图无框 → 跳过（或保留空标签？保留空标签以保持图片）
            pass  # 仍然保留（空标签的图也有训练价值）

        by_class[cls_name].append((lbl_path, stem, new_cls_id, new_lines))

    # 统计
    print(f"\n各类标注统计:")
    total = 0
    for cls_name in CLASS_MAP:
        items = by_class[cls_name]
        n_boxed = sum(1 for _, _, _, lines in items if lines)
        total += len(items)
        print(f"  {cls_name}: {len(items)} 张图, {n_boxed} 张有框")

    print(f"  总计: {total} 张")
    if unknown:
        print(f"  未识别: {len(unknown)} ({unknown[:5]}...)")

    # 分层 split + 写文件
    print(f"\n切分 train/val (val_ratio={VAL_RATIO})...")

    for cls_name in CLASS_MAP:
        items = by_class[cls_name]
        random.shuffle(items)
        n_val = max(1, int(len(items) * VAL_RATIO))

        for i, (lbl_path, stem, new_cls_id, new_lines) in enumerate(items):
            split = "val" if i < n_val else "train"

            # 找对应图片
            img_src = None
            for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                candidate = GOLDEN_SRC / "images" / f"{stem}{ext}"
                if candidate.exists():
                    img_src = candidate
                    break

            if img_src is None:
                print(f"  [WARN] 图片缺失: {stem}")
                continue

            # 输出名
            out_stem = f"{cls_name}_{img_src.stem}"
            out_img = GOLDEN_OUT / split / "images" / f"{out_stem}{img_src.suffix}"
            out_lbl = GOLDEN_OUT / split / "labels" / f"{out_stem}.txt"

            # 写标签
            if new_lines:
                out_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            else:
                out_lbl.write_text("", encoding="utf-8")  # 空标签（健康图或无病灶图）

            # 复制图片
            if not out_img.exists():
                shutil.copy2(img_src, out_img)

            fixed_count += 1

    # 统计输出
    train_imgs = len(list((GOLDEN_OUT / "train" / "images").glob("*")))
    val_imgs = len(list((GOLDEN_OUT / "val" / "images").glob("*")))
    train_lbls = len(list((GOLDEN_OUT / "train" / "labels").glob("*")))
    val_lbls = len(list((GOLDEN_OUT / "val" / "labels").glob("*")))

    print(f"\n输出统计:")
    print(f"  Train: {train_imgs} 图片, {train_lbls} 标签")
    print(f"  Val:   {val_imgs} 图片, {val_lbls} 标签")
    print(f"  Total: {train_imgs + val_imgs}")

    # 每类框数
    box_count = Counter()
    for split in ["train", "val"]:
        for lbl in (GOLDEN_OUT / split / "labels").glob("*.txt"):
            for line in lbl.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    box_count[int(line.split()[0])] += 1

    print(f"\n各类框数:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}: {box_count.get(i, 0)}")
    print(f"  总框: {sum(box_count.values())}")

    # 写 classes.txt
    (GOLDEN_OUT / "classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")

    # 写 data.yaml
    yaml_content = f"""path: {GOLDEN_OUT}
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
    (GOLDEN_OUT / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"\n✅ 转换完成!")
    print(f"  输出目录: {GOLDEN_OUT}")
    print(f"  data.yaml: {GOLDEN_OUT / 'data.yaml'}")
    print(f"  classes.txt: {GOLDEN_OUT / 'classes.txt'}")


if __name__ == "__main__":
    main()
