"""
筛选黄金标注候选图 — 各类均衡抽样, 优先高清
输出: golden_set/images/
"""
import random
import shutil
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ========== 配置 ==========
ARCHIVE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\archive\Wheat_Disease\train")
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\golden_set")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

CLASSES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Mildew", "Fusarium Head Blight", "Healthy Wheat",
]

# 每类最多选几张 (Mildew/Leaf Blight 资源少，取25)
MAX_PER_CLASS = 25
MIN_SHORT_SIDE = 640   # 只选高清图
SEED = 42

random.seed(SEED)


def collect_candidates():
    """收集每类 >=640px 的图片，按分辨率分层"""
    candidates = defaultdict(list)

    for cls_name in CLASSES:
        cls_dir = ARCHIVE_DIR / cls_name
        if not cls_dir.exists():
            continue

        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                short = min(w, h)
                if short >= MIN_SHORT_SIDE:
                    candidates[cls_name].append((img_path, w, h, short))
            except Exception:
                continue

    return candidates


def select_diverse(candidates, n):
    """分层抽样：按分辨率分3档，每档均匀取"""
    if len(candidates) <= n:
        return candidates

    # 按短边分档
    sorted_cands = sorted(candidates, key=lambda x: x[3])  # by short side
    third = len(sorted_cands) // 3
    low = sorted_cands[:third]
    mid = sorted_cands[third:2*third]
    high = sorted_cands[2*third:]

    selected = []
    per_bin = max(1, n // 3)

    for bin_cands in [low, mid, high]:
        random.shuffle(bin_cands)
        selected.extend(bin_cands[:per_bin])

    # 不够的补足
    if len(selected) < n:
        remaining = [c for c in candidates if c not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


def main():
    print("=" * 60)
    print("黄金标注集筛选")
    print(f"每类最多: {MAX_PER_CLASS} 张, 短边 >= {MIN_SHORT_SIDE}px")
    print("=" * 60)

    # 清空旧数据
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates()
    total_selected = 0

    print(f"\n{'Class':<22} {'候选':>6} {'选中':>6}")
    print("-" * 38)

    for cls_name in CLASSES:
        cands = candidates[cls_name]
        n_cands = len(cands)
        n_select = min(MAX_PER_CLASS, n_cands)
        selected = select_diverse(cands, n_select)

        for img_path, w, h, short in selected:
            out_name = f"{cls_name}_{img_path.stem}.jpg"
            # 转换RGBA/CMYK等
            img = Image.open(img_path)
            if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                img = img.convert('RGB')
            img.save(IMAGES_DIR / out_name, quality=95)
            total_selected += 1

        print(f"{cls_name:<22} {n_cands:>6} {len(selected):>6}")

    print("-" * 38)
    print(f"{'合计':<22} {'':>6} {total_selected:>6}")

    # 写入 classes.txt
    (OUTPUT_DIR / "classes.txt").write_text("\n".join(CLASSES), encoding='utf-8')

    print(f"\n图片已保存到: {IMAGES_DIR}")
    print(f"标签将保存到: {LABELS_DIR}")
    print(f"\n运行标注工具: python label_tool.py")
    print(f"  - 鼠标拖拽画框")
    print(f"  - 数字键 0-7 切换类别")
    print(f"  - N: 下一张  D: 删除框  Q: 退出")


if __name__ == "__main__":
    main()
