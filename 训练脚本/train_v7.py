"""
V7: V5微调 + 726张纯净数据（精标+审核+锈病终审）
"""
from ultralytics import YOLO
from pathlib import Path

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
DATA = PROJECT / "wheat_v7_clean"
PRETRAIN = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"
NAME = "wheat_7class_v7"

data_yaml = DATA / "data.yaml"
data_yaml.write_text(f"""path: {DATA}
train: train/images
val: val/images
nc: 7
names:
  0: Brown_Rust
  1: Yellow_Rust
  2: Black_Rust
  3: Septoria
  4: Powdery_Mildew
  5: Fusarium_Head_Blight
  6: Healthy_Wheat
""", encoding="utf-8")

print(f"V7: 从V5微调，726张纯净标签")
print(f"预训练: {PRETRAIN}")
print(f"数据: {DATA}")

if __name__ == '__main__':
    model = YOLO(str(PRETRAIN))

    results = model.train(
        data=str(data_yaml),
        epochs=200,
        imgsz=640,
        batch=16,           # 小数据集用更小batch
        device=0,
        workers=4,
        lr0=0.0002,         # 更低lr保护V5权重
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        hsv_h=0.02,         # 更强增强补偿小数据
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=15.0,
        translate=0.15,
        scale=0.6,
        shear=5.0,
        perspective=0.0001,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        patience=50,
        close_mosaic=30,
        cos_lr=True,
        project=str(PROJECT / "runs"),
        name=NAME,
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    print(f"\nDone. Best: {results.save_dir}")
