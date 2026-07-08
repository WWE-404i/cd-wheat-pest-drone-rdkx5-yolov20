"""
V9种子: yolo26n从头训, 726张新标准标签, 用于迭代伪标
"""
from ultralytics import YOLO
from pathlib import Path

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
DATA = PROJECT / "wheat_v9_seed"
NAME = "wheat_7class_v9"

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

print(f"V9: yolo26n从头训, 726张种子")
print(f"数据: {DATA}")

if __name__ == '__main__':
    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=15.0,
        translate=0.15,
        scale=0.6,
        shear=5.0,
        perspective=0.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        patience=30,
        close_mosaic=20,
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
