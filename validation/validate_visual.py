"""
V5模型可视化验证：在精标验证集上跑推理，生成对比图
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"
GOLDEN = PROJECT / "wheat_disease_golden"
OUT = PROJECT / "validation_samples"

CLASS_NAMES = ["Brown_Rust","Yellow_Rust","Black_Rust","Septoria",
               "Powdery_Mildew","Fusarium_Head_Blight","Healthy_Wheat"]

# 清理旧结果
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()

# 在精标验证集上跑验证（带标签对比）
model = YOLO(str(MODEL))

results = model.val(
    data=str(GOLDEN / "data.yaml"),
    imgsz=640,
    batch=16,
    device=0,
    split="val",
    plots=True,
    save_json=False,
)

# 复制验证结果图
plots_dir = PROJECT / "runs" / "wheat_7class_v5"
for png in sorted(plots_dir.glob("*.png")):
    shutil.copy2(png, OUT / png.name)

# 在几张样本图上推理，保存预测结果
print("\n=== 推理样本图 ===")
val_img_dir = GOLDEN / "val" / "images"
if val_img_dir.exists():
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)

    sample_imgs = sorted(val_img_dir.glob("*.jpg"))[:20]
    for img in sample_imgs:
        model.predict(
            source=str(img), imgsz=640, device=0,
            conf=0.35, save=True, project=str(pred_dir), name=".",
            exist_ok=True,
        )

    # 移动预测结果
    for p in (pred_dir / ".").glob("*.jpg"):
        if p.name in [x.name for x in sample_imgs]:
            shutil.move(str(p), str(pred_dir / p.name))

print(f"\n完成 => {OUT}")
print(f"验证集指标: mAP50={results.box.map50:.4f}  mAP50-95={results.box.map:.4f}")
if hasattr(results.box, 'mr'):
    print(f"Recall: {results.box.mr:.4f}  Precision: {results.box.mp:.4f}")
