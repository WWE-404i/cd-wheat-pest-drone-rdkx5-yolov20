"""检查审核图片中 class 0 和 2 的分布"""
from pathlib import Path

REVIEWED = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_pseudo_v5_reviewed\labels")

has_both = []  # 同时有0和2
has_0_only = []
has_2_only = []
no_0_2 = []

for lbl in sorted(REVIEWED.glob("*.txt")):
    classes = set()
    for line in lbl.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if parts:
            classes.add(int(parts[0]))

    if 0 in classes and 2 in classes:
        has_both.append(lbl.stem)
    elif 0 in classes:
        has_0_only.append(lbl.stem)
    elif 2 in classes:
        has_2_only.append(lbl.stem)
    else:
        no_0_2.append(lbl.stem)

print(f"总审核图片: {len(list(REVIEWED.glob('*.txt')))}")
print(f"同时有叶锈(0)和秆锈(2): {len(has_both)} 张 -- 最可能混淆!")
print(f"只有叶锈(0): {len(has_0_only)} 张")
print(f"只有秆锈(2): {len(has_2_only)} 张")
print(f"都没有: {len(no_0_2)} 张")
if has_both[:10]:
    print(f"\n同时有0和2的前10张: {has_both[:10]}")
if has_0_only[:5]:
    print(f"只有0的前5张: {has_0_only[:5]}")
if has_2_only[:5]:
    print(f"只有2的前5张: {has_2_only[:5]}")
