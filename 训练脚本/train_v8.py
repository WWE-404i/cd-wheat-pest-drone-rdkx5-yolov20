"""
V8: V5微调, 2800张混合(锈病人工审 + 非锈病V5伪标)
"""
from ultralytics import YOLO
from pathlib import Path

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
DATA = PROJECT / "wheat_v8_hybrid"
PRETRAIN = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"
NAME = "wheat_7class_v8"

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

print(f"V8: 从V5微调, 2800张混合(锈病人工审)")
print(f"预训练: {PRETRAIN}")

if __name__ == '__main__':
    model = YOLO(str(PRETRAIN))

    results = model.train(
        data=str(data_yaml),
        epochs=150,
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
        patience=40,
        close_mosaic=25,
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
