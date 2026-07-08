"""
从 image_audit/keep 中随机挑选待标注图片，补足每类到 50 张
输出: to_label/{class_name}/ 和 label_tool_v2.py 配套
"""
import random
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
KEEP_DIR = PROJECT / "image_audit" / "keep_短边大于等于640"
GOLDEN_DIR = PROJECT / "golden_set_7class"
OUTPUT = PROJECT / "to_label"

# 目标：每类 50 张
TARGET_PER_CLASS = 50

# 7 类对应 keep 文件夹名
CLASS_FOLDERS = {
    "Brown Rust": "Brown Rust",
    "Yellow Rust": "Yellow Rust",
    "Black Rust": "Black Rust",
    "Septoria": "Septoria",
    "Mildew": "Mildew",
    "Fusarium Head Blight": "Fusarium Head Blight",
    "Healthy Wheat": "Healthy Wheat",
}

SEED = 42
random.seed(SEED)


def count_existing():
    """统计 golden_set_7class 中各类已有标注数"""
    counts = defaultdict(int)
    for split in ["train", "val"]:
        lbl_dir = GOLDEN_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            for cls_name in CLASS_FOLDERS:
                if lbl.stem.startswith(cls_name + "_"):
                    # 检查是否有框（非空标注）
                    content = lbl.read_text().strip()
                    if content:
                        counts[cls_name] += 1
                    break
    return counts


def main():
    existing = count_existing()
    print("golden_set_7class 已有标注（有框图）:")
    total_existing = 0
    for cls_name in CLASS_FOLDERS:
        n = existing.get(cls_name, 0)
        total_existing += n
        need = max(0, TARGET_PER_CLASS - n)
        print(f"  {cls_name}: {n} 张, 还需 {need} 张")
    print(f"  总计: {total_existing} 张")

    # 准备输出
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    total_selected = 0

    for cls_name, folder_name in CLASS_FOLDERS.items():
        src_dir = KEEP_DIR / folder_name
        need = max(0, TARGET_PER_CLASS - existing.get(cls_name, 0))

        if not src_dir.exists() or need <= 0:
            if need <= 0:
                print(f"  {cls_name}: 已够，跳过")
            else:
                print(f"  {cls_name}: 图片来源缺失! ({src_dir})")
            continue

        # 收集候选
        candidates = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        if not candidates:
            print(f"  {cls_name}: 无候选图片!")
            continue

        random.shuffle(candidates)
        selected = candidates[:need]

        # 复制
        out_dir = OUTPUT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in selected:
            shutil.copy2(img_path, out_dir / img_path.name)

        total_selected += len(selected)
        print(f"  {cls_name}: 选中 {len(selected)}/{len(candidates)} → {out_dir}")

    print(f"\n共选出 {total_selected} 张")
    print(f"输出: {OUTPUT}")
    print(f"\n运行标注: python label_tool_v2.py")


if __name__ == "__main__":
    main()
