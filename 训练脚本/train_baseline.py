"""
YOLOv26n 小麦病变检测 Baseline 训练
数据集: wheat_yolo_train (GD/SAM 标注)
7类: Brown Rust, Yellow Rust, Black Rust, Septoria, Mildew, Fusarium Head Blight, Healthy Wheat
"""

import os
from pathlib import Path
from ultralytics import YOLO

# ========== 配置 ==========
PROJECT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
DATA_YAML = PROJECT_DIR / "wheat_yolo_train" / "data.yaml"
MODEL_NAME = "yolo26n.pt"  # 自动下载预训练权重
PROJECT = str(PROJECT_DIR / "runs" / "wheat_baseline")
NAME = "train7"

# 训练参数
EPOCHS = 200
IMSZ = 640
BATCH = 40  # RTX 5060 8G, 吃满显存
DEVICE = 0  # GPU

def main():
    print("=" * 60)
    print("YOLOv26n 小麦病变检测 Baseline 训练")
    print("=" * 60)
    print(f"数据集: {DATA_YAML}")
    print(f"模型: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"分辨率: {IMSZ}")
    print(f"Batch: {BATCH}")
    print(f"输出: {PROJECT}/{NAME}")
    print("=" * 60)

    # 加载模型
    model = YOLO(MODEL_NAME)

    # 训练
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        # 优化参数
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        # 数据增强 (V3 配置)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # 关闭 mosaic 早停
        close_mosaic=30,
        # 保存
        save=True,
        save_period=10,
        # 验证
        val=True,
        # 其他
        workers=0,  # Windows multiprocessing 会卡死, 用0
        pretrained=True,
        amp=True,
        verbose=True,
    )

    print("\n训练完成!")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")

    # 验证
    print("\n运行验证...")
    metrics = model.val(data=str(DATA_YAML), split="val", device=DEVICE)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == '__main__':
    main()
