"""
V1 手工标注模型训练
数据: wheat_7class_manual (383张, 790框, 7类)
用途: 训一个小模型做自动预标注
"""
from ultralytics import YOLO
from pathlib import Path


def main():
    PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
    DATA_YAML = PROJECT / "wheat_7class_manual" / "data.yaml"

    # 加载预训练权重
    model = YOLO(PROJECT / "yolo26n.pt")

    # 训练: 数据少，开 mosaic 增强 + 多训一些 epoch
    results = model.train(
        data=str(DATA_YAML),
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        # 项目设置
        project=str(PROJECT / "runs"),
        name="wheat_7class_manual_v1",

        # 优化器
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,

        # 数据增强 (小数据集加强增强)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,

        # 关闭 close_mosaic (小数据集不关)
        close_mosaic=0,

        # 验证
        val=True,
        save=True,
        save_period=50,

        # 早停
        patience=100,
    )


if __name__ == '__main__':
    main()
