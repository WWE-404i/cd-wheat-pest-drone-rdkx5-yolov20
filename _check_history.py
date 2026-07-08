"""查看三轮审核历史"""
import json, os
from pathlib import Path
from datetime import datetime

BASE = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")

checkpoints = [
    "florence_checkpoint.json",
    "manual_label_checkpoint.json",
    "manual_label_checkpoint_v2.json",
    "review_checkpoint.json",
]

for cp in checkpoints:
    fp = BASE / cp
    if fp.exists():
        mtime = datetime.fromtimestamp(os.path.getmtime(fp))
        try:
            data = json.loads(fp.read_text())
        except:
            print(f"{cp}: {mtime.strftime('%Y-%m-%d %H:%M:%S')}, 读取失败")
            continue
        n = len(data) if isinstance(data, list) else "not a list"
        print(f"{cp}: {mtime}, {n} entries")
        if isinstance(data, list) and len(data) > 0:
            print(f"  First: {data[0]}")
            print(f"  Last:  {data[-1]}")
        elif isinstance(data, dict):
            keys = list(data.keys())
            print(f"  Keys (first 5): {keys[:5]}")
        print()

# 审核标签时间分簇
print("=== 审核标签文件时间分布 ===")
lbl_dir = BASE / "wheat_pseudo_v5_reviewed" / "labels"
if lbl_dir.exists():
    times = []
    for lbl in lbl_dir.glob("*.txt"):
        t = os.path.getmtime(lbl)
        times.append((t, lbl.stem))

    if times:
        times.sort()
        print(f"总数: {len(times)}")
        print(f"最早: {datetime.fromtimestamp(times[0][0]).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"最晚: {datetime.fromtimestamp(times[-1][0]).strftime('%Y-%m-%d %H:%M:%S')}")

        # 分簇
        clusters = []
        current_start = times[0][0]
        current_end = times[0][0]
        current_stems = [times[0][1]]
        for t, stem in times[1:]:
            if t - current_end > 1800:  # 30 min gap
                clusters.append((current_start, current_end, current_stems))
                current_start = t
                current_stems = [stem]
            else:
                current_stems.append(stem)
            current_end = t
        clusters.append((current_start, current_end, current_stems))

        print(f"\n时间分簇 ({len(clusters)} 轮):")
        for i, (s, e, stems) in enumerate(clusters):
            print(f"  第{i+1}轮: {datetime.fromtimestamp(s).strftime('%m-%d %H:%M')} ~ "
                  f"{datetime.fromtimestamp(e).strftime('%m-%d %H:%M')}, {len(stems)}张")
            if len(stems) <= 5:
                print(f"    图片: {stems}")

# 审核与断点交叉对比
print("\n=== 各断点覆盖对比 ===")
reviewed_lbl_dir = BASE / "wheat_pseudo_v5_reviewed" / "labels"
reviewed_all = set()
if reviewed_lbl_dir.exists():
    for lbl in reviewed_lbl_dir.glob("*.txt"):
        reviewed_all.add(lbl.stem)

for cp in checkpoints:
    fp = BASE / cp
    if fp.exists():
        try:
            data = json.loads(fp.read_text())
        except:
            continue
        if isinstance(data, list) and len(data) > 0:
            cp_set = set(data)
            overlap = cp_set & reviewed_all
            only_cp = cp_set - reviewed_all
            only_rev = reviewed_all - cp_set
            print(f"{cp}: 断点{len(cp_set)}张, 与审核标签重叠{len(overlap)}张, "
                  f"仅在断点{len(only_cp)}张, 仅审核标签{len(only_rev)}张")
