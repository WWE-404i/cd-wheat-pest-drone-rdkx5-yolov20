"""复查前后对比"""
from pathlib import Path
from datetime import datetime
import os

CLS = ["Brown_Rust(0)", "Yellow_Rust(1)", "Black_Rust(2)", "Septoria(3)",
       "Powdery_Mildew(4)", "Fusarium_HB(5)", "Healthy(6)"]

REVIEWED = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_pseudo_v5_reviewed\labels")

# 统计各图片各类框数
now = {}
recent = []  # 刚改过的
for lbl in sorted(REVIEWED.glob("*.txt")):
    counts = {}
    for line in lbl.read_text().strip().splitlines():
        if not line.strip():
            continue
        c = int(line.split()[0])
        counts[c] = counts.get(c, 0) + 1
    now[lbl.stem] = counts
    mtime = datetime.fromtimestamp(os.path.getmtime(lbl))
    # 今天改的
    if mtime.date() == datetime.now().date():
        recent.append((lbl.stem, counts, mtime))

total_now = {}
for cts in now.values():
    for c, n in cts.items():
        total_now[c] = total_now.get(c, 0) + n

print("=== 当前审核标签统计 ===")
print(f"图片数: {len(now)}")
for c in range(7):
    print(f"  {CLS[c]}: {total_now.get(c, 0)} 框")

print(f"\n=== 今日修改: {len(recent)} 张 ===")
for stem, cts, t in sorted(recent, key=lambda x: x[2]):
    parts = [f"{CLS[c]}:{n}" for c, n in sorted(cts.items())]
    print(f"  {stem}: {', '.join(parts)}")

# 检查是否有图片同时有0和2（潜在冲突）
print(f"\n=== 仍有0和2共存的图片 ===")
both = []
for stem, cts in now.items():
    if 0 in cts and 2 in cts:
        both.append(stem)
        print(f"  {stem}: {dict(cts)}")
print(f"共 {len(both)} 张")
