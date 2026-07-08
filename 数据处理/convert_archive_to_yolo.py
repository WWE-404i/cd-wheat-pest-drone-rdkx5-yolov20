"""
将 archive/Wheat_Disease 分类数据集转为 YOLO 检测格式
策略: 每张图一个中心缩框（60%图幅），类别=文件夹名
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random

# ========== 配置 ==========
ARCHIVE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\archive\Wheat_Disease")
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_disease_8class")

# 8个核心类别
SELECTED_CLASSES = [
    "Brown Rust",           # 褐锈病 - 3248 train
    "Yellow Rust",          # 黄锈病 - 2379 train
    "Black Rust",           # 黑锈病 - 933 train
    "Septoria",             # 壳针孢叶斑病 - 1140 train
    "Mildew",               # 白粉病 - 884 train
    "Fusarium Head Blight", # 赤霉病 - 1020 train
    "Leaf Blight",          # 叶枯病 - 673 train
    "Healthy Wheat",        # 健康小麦 - 2907 train
]

# 框占图幅的比例（中心缩框）
BOX_RATIO = 0.60  # 60% 图幅

# 验证集比例（从训练集分出，因为原val也是分类格式没标注）
VAL_RATIO = 0.15

# 随机种子
SEED = 42


def create_yolo_box(img_w, img_h, box_ratio=BOX_RATIO):
    """生成中心缩框，返回 YOLO 格式 [x_center, y_center, width, height] 归一化"""
    w = box_ratio
    h = box_ratio
    x = 0.5  # center
    y = 0.5  # center
    return [x, y, w, h]


def process_class(class_name, class_id, train_dir, out_images, out_labels, file_list):
    """处理一个类别，生成缩框标注"""
    src_dir = ARCHIVE_DIR / "train" / class_name
    if not src_dir.exists():
        print(f"  [SKIP] {class_name}: directory not found")
        return 0

    images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG")) + list(src_dir.glob("*.png"))
    if not images:
        print(f"  [SKIP] {class_name}: no images")
        return 0

    count = 0
    for img_path in images:
        try:
            # 获取图像尺寸
            img = Image.open(img_path)
            orig_w, orig_h = img.size

            # 生成中心缩框
            box = create_yolo_box(orig_w, orig_h)

            # 文件名
            stem = img_path.stem
            out_img_name = f"{class_name}_{stem}.jpg"
            out_label_name = f"{class_name}_{stem}.txt"

            # 转成 jpg 统一格式
            if img_path.suffix.lower() == '.png':
                img = img.convert('RGB')

            # 保存
            img.save(out_images / out_img_name, quality=95)

            with open(out_labels / out_label_name, 'w') as f:
                f.write(f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

            file_list.append((out_img_name, out_label_name))
            count += 1

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")

    return count


def main():
    print("=" * 60)
    print("Archive → YOLO 检测格式转换")
    print(f"策略: 中心缩框 {BOX_RATIO*100:.0f}% 图幅, {len(SELECTED_CLASSES)} 类")
    print("=" * 60)

    # 创建输出目录
    train_img_dir = OUTPUT_DIR / "train" / "images"
    train_lbl_dir = OUTPUT_DIR / "train" / "labels"
    val_img_dir = OUTPUT_DIR / "val" / "images"
    val_lbl_dir = OUTPUT_DIR / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    all_train_files = []
    all_val_files = []

    for i, class_name in enumerate(SELECTED_CLASSES):
        print(f"\n处理: {class_name} (class_id={i})")

        # 先收集所有文件到一个临时列表
        src_dir = ARCHIVE_DIR / "train" / class_name
        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.JPG")) + list(src_dir.glob("*.png"))

        if not images:
            continue

        # 随机划分 train/val
        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_RATIO))
        train_imgs = images[n_val:]
        val_imgs = images[:n_val]

        # 处理训练集
        count_train = 0
        for img_path in train_imgs:
            try:
                img = Image.open(img_path)
                orig_w, orig_h = img.size
                box = create_yolo_box(orig_w, orig_h)

                stem = img_path.stem
                out_img_name = f"{class_name}_{stem}.jpg"
                out_label_name = f"{class_name}_{stem}.txt"

                if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                    img = img.convert('RGB')

                img.save(train_img_dir / out_img_name, quality=95)
                with open(train_lbl_dir / out_label_name, 'w') as f:
                    f.write(f"{i} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                all_train_files.append(out_img_name)
                count_train += 1
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

        # 处理验证集
        count_val = 0
        for img_path in val_imgs:
            try:
                img = Image.open(img_path)
                orig_w, orig_h = img.size
                box = create_yolo_box(orig_w, orig_h)

                stem = img_path.stem
                out_img_name = f"{class_name}_{stem}.jpg"
                out_label_name = f"{class_name}_{stem}.txt"

                if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                    img = img.convert('RGB')

                img.save(val_img_dir / out_img_name, quality=95)
                with open(val_lbl_dir / out_label_name, 'w') as f:
                    f.write(f"{i} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                all_val_files.append(out_img_name)
                count_val += 1
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

        print(f"  Train: {count_train}, Val: {count_val}")

    # 生成 data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    # 使用绝对路径
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images
nc: {len(SELECTED_CLASSES)}
names: {SELECTED_CLASSES}
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # 汇总
    print(f"\n{'='*60}")
    print(f"转换完成!")
    print(f"  训练集: {len(all_train_files)} 张")
    print(f"  验证集: {len(all_val_files)} 张")
    print(f"  类别数: {len(SELECTED_CLASSES)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  配置文件: {yaml_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
