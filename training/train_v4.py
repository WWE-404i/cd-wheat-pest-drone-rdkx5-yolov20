"""
V4: 534张精标训练
- 从 V3 best.pt 微调（mAP=0.772 基础）
- 小数据集 → 强增强防过拟合
- 可选: 先跑伪标注扩充再训练
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
GOLDEN_DIR = PROJECT / "wheat_disease_golden"
V3_WEIGHTS = PROJECT / "runs" / "wheat_7class_v33" / "weights" / "best.pt"
SEED = 42
random.seed(SEED)


def train():
    model = YOLO(str(V3_WEIGHTS))  # 从V3微调，不是从头训练

    model.train(
        data=str(GOLDEN_DIR / "data.yaml"),
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        # 小数据集优化器设置
        optimizer="AdamW",
        lr0=0.0005,       # 微调用更低lr（V3是0.001）
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,

        # 强数据增强（小数据集补偿）
        hsv_h=0.02,        # 色调
        hsv_s=0.7,         # 饱和度
        hsv_v=0.4,         # 明度
        degrees=15.0,      # 旋转（加大）
        translate=0.15,    # 平移
        scale=0.6,         # 缩放（加大）
        shear=5.0,         # 剪切（加大）
        perspective=0.001, # 透视
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,        # mixup（加大）
        copy_paste=0.1,    # copy-paste增强
        close_mosaic=25,   # 最后25轮关mosaic

        # 验证与保存
        val=True,
        save=True,
        save_period=20,
        patience=30,       # 小数据集给更多耐心

        project=str(PROJECT / "runs"),
        name="wheat_7class_v4",
    )


if __name__ == "__main__":
    train()
