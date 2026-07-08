"""
V6b训练: V5伪标注 + 审核替换，干净策略
"""
from ultralytics import YOLO
from pathlib import Path

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
DATA = PROJECT / "wheat_v6b_combined"
PRETRAIN = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"  # 从V5继续
NAME = "wheat_7class_v6b"

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

print(f"V6b: {PRETRAIN} 微调")
print(f"数据: {DATA}")

if __name__ == '__main__':
    model = YOLO(str(PRETRAIN))

    results = model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=640,
        batch=32,
        device=0,
        workers=4,
        lr0=0.0005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        patience=50,
        close_mosaic=30,
        cos_lr=True,
        dropout=0.0,
        project=str(PROJECT / "runs"),
        name=NAME,
        exist_ok=True,
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )

    print(f"\nDone. Best: {results.save_dir}")
