"""检查锈病审核结果"""
import json, os
from pathlib import Path
from datetime import datetime
from collections import Counter

BASE = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
RUST_DIR = BASE / "rust_review" / "labels"
CKP = BASE / "review_checkpoint_v4.json"

CLS = ["Brown_Rust(0)","Yellow_Rust(1)","Black_Rust(2)","Septoria(3)",
       "Powdery_Mildew(4)","Fusarium_HB(5)","Healthy(6)"]

# 断点
if CKP.exists():
    done = json.loads(CKP.read_text())
    print(f"断点完成: {len(done)} 张")
else:
    done = set()
    print("无断点")

# 当前标签统计
total_cls = Counter()
img_cls = {}  # stem -> set of classes
img_counts = {}  # stem -> counts
recent = []

for lbl in sorted(RUST_DIR.glob("*.txt")):
    classes = set()
    counts = Counter()
    for line in lbl.read_text().strip().splitlines():
        if not line.strip():
            continue
        c = int(line.split()[0])
        classes.add(c)
        counts[c] += 1
        total_cls[c] += 1
    img_cls[lbl.stem] = classes
    img_counts[lbl.stem] = counts

    mtime = datetime.fromtimestamp(os.path.getmtime(lbl))
    today = datetime.now().date()
    if mtime.date() == today:
        recent.append(lbl.stem)

print(f"\n标签文件总数: {len(list(RUST_DIR.glob('*.txt')))}")
print(f"今日修改: {len(recent)} 张")
print(f"\n各类框数:")
for c in range(7):
    print(f"  {CLS[c]}: {total_cls.get(c,0)}")

# 多种锈病共存
multi_rust = []
for stem, classes in img_cls.items():
    rust = classes & {0,1,2}
    if len(rust) >= 2:
        multi_rust.append((stem, rust, img_counts[stem]))

print(f"\n多种锈病共存: {len(multi_rust)} 张")
for stem, rusts, cts in sorted(multi_rust):
    rust_names = [CLS[c] for c in rusts]
    all_cts = {CLS[c]: n for c, n in cts.items()}
    print(f"  {stem}: 锈病={rust_names}  全部={all_cts}")

# 空标签
empty = [s for s, cts in img_counts.items() if len(cts) == 0]
print(f"\n空标签(全删): {len(empty)} 张")
if empty:
    for s in empty:
        print(f"  {s}")
