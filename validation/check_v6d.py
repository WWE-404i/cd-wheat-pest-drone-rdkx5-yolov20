"""快速检查V6d训练指标"""
import csv
p = r"D:\python\程序\mnist-YOLO\小麦病变识别模型\runs\wheat_7class_v6d\results.csv"
try:
    with open(p) as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda x: float(x["metrics/mAP50(B)"]))
    last = rows[-1]
    print(f"总epoch: {len(rows)}")
    print(f"最佳 ep {best['epoch']}: mAP50={float(best['metrics/mAP50(B)']):.4f}  R={float(best['metrics/recall(B)']):.4f}  P={float(best['metrics/precision(B)']):.4f}  mAP50-95={float(best['metrics/mAP50-95(B)']):.4f}")
    print(f"最后 ep {last['epoch']}: mAP50={float(last['metrics/mAP50(B)']):.4f}  R={float(last['metrics/recall(B)']):.4f}  P={float(last['metrics/precision(B)']):.4f}")
    print("\n最近5:")
    for r in rows[-5:]:
        flag = " <-- BEST" if r["epoch"] == best["epoch"] else ""
        print(f"  ep{r['epoch']:>3s}: mAP50={float(r['metrics/mAP50(B)']):.4f}  R={float(r['metrics/recall(B)']):.4f}  P={float(r['metrics/precision(B)']):.4f}{flag}")
except FileNotFoundError:
    print("还没生成 results.csv")
