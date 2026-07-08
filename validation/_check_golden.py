"""检查精标534里的潜在混淆"""
from pathlib import Path

GOLDEN = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_disease_golden")
CLS = ["Brown_Rust(0)","Yellow_Rust(1)","Black_Rust(2)","Septoria(3)",
       "Powdery_Mildew(4)","Fusarium_HB(5)","Healthy(6)"]

# 统计每个图片的类别
both_0_2 = []
has_0 = []
has_2 = []
all_stats = {}

for split in ["train", "val"]:
    lbl_dir = GOLDEN / split / "labels"
    if not lbl_dir.exists():
        continue
    for lbl in sorted(lbl_dir.glob("*.txt")):
        classes = set()
        counts = {}
        for line in lbl.read_text().strip().splitlines():
            if not line.strip(): continue
            c = int(line.split()[0])
            classes.add(c)
            counts[c] = counts.get(c, 0) + 1

        stem = lbl.stem
        all_stats[stem] = counts

        if 0 in classes and 2 in classes:
            both_0_2.append(stem)
        elif 0 in classes:
            has_0.append(stem)
        elif 2 in classes:
            has_2.append(stem)

print(f"精标总数: {len(all_stats)}")
print(f"同时有叶锈(0)和秆锈(2): {len(both_0_2)} 张")
print(f"只有叶锈(0): {len(has_0)} 张")
print(f"只有秆锈(2): {len(has_2)} 张")

# 总数统计
total_cls = {}
for cts in all_stats.values():
    for c, n in cts.items():
        total_cls[c] = total_cls.get(c, 0) + n
print(f"\n各类框数:")
for c in range(7):
    print(f"  {CLS[c]}: {total_cls.get(c,0)}")

# 列出同时有0和2的
if both_0_2:
    print(f"\n同时有0和2的图片 ({len(both_0_2)}张):")
    for s in both_0_2[:10]:
        print(f"  {s}: {all_stats[s]}")

# 从 Black_Rust 文件夹来的图片（原Kaggle的Black_Rust文件夹）
print(f"\n=== 按原文件夹分布 ===")
from collections import Counter
folder_dist = Counter()
for s in all_stats:
    # 金标文件夹名通常映射原类别
    parts = s.split('_')
    folder_dist[s] = all_stats[s]

# 简化：统计文件名前缀
prefixes = Counter()
for s in all_stats:
    # 提取前缀如 "Black Rust", "Brown Rust", "Septoria"等
    for prefix in ["Black Rust", "Brown Rust", "Yellow Rust", "Septoria",
                   "Powdery Mildew", "Fusarium Head Blight", "Healthy Wheat"]:
        if s.startswith(prefix):
            prefixes[prefix] += 1
            break

print("按原Kaggle类别分布:")
for k, v in prefixes.most_common():
    print(f"  {k}: {v}张")
