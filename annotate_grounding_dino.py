"""
小麦病害 Grounding DINO 标注脚本
使用开放词汇目标检测，给定文字描述直接输出 bbox
专为检测任务设计，不走 VLM 弯路
"""

import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# ========== 配置 ==========
ARCHIVE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\archive\Wheat_Disease")
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_detection_gd")

# 选用核心病害类
SELECTED_CLASSES = [
    "Brown Rust",
    "Yellow Rust",
    "Septoria",
    "Mildew",
    "Black Rust",
    "Healthy Wheat",
]

# 测试阶段每类2张
SAMPLES_PER_CLASS = 2

# 文字描述映射 — 用于 Grounding DINO 的 text prompt
DISEASE_PROMPTS = {
    "Brown Rust": "brown rust spots on wheat leaf . orange brown lesions",
    "Yellow Rust": "yellow rust pustules on wheat leaf . yellow orange stripes",
    "Septoria": "septoria leaf blotch on wheat . brown spots with yellow halo",
    "Mildew": "powdery mildew on wheat leaf . white powdery patches",
    "Black Rust": "black rust on wheat leaf . dark brown black lesions",
    "Healthy Wheat": "green wheat leaf . healthy plant",
}

CLASS_NAMES_CN = {
    "Brown Rust": "褐锈病",
    "Yellow Rust": "黄锈病",
    "Septoria": "壳针孢叶斑病",
    "Mildew": "白粉病",
    "Black Rust": "黑锈病",
    "Healthy Wheat": "健康小麦",
}


def load_grounding_dino():
    """加载 Grounding DINO"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"

    print(f"Loading {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
    ).to("cuda")

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor


def detect_disease(model, processor, image_path, text_prompt, box_threshold=0.25, text_threshold=0.2):
    """用 Grounding DINO 检测病害区域"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Grounding DINO 需要用固定格式的 prompt: "a . b . c"
    # 每个 . 分隔的短语会被独立匹配

    inputs = processor(images=img, text=text_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    # 后处理
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(orig_h, orig_w)]
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"] if "labels" in results else [text_prompt] * len(boxes)

    return boxes, scores, labels


def boxes_to_yolo(boxes, img_w, img_h):
    """转换 [x1,y1,x2,y2] 到 YOLO [x_center, y_center, width, height] 归一化"""
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        # 裁剪到 [0,1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0.01, min(1, width))
        height = max(0.01, min(1, height))
        yolo_boxes.append([x_center, y_center, width, height])
    return yolo_boxes


def save_yolo_label(label_path, boxes, class_id):
    """保存 YOLO 格式标注"""
    if not boxes:
        return
    with open(label_path, 'w') as f:
        for box in boxes:
            x, y, w, h = box
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    print("=" * 60)
    print("小麦病害标注 — Grounding DINO (开放词汇目标检测)")
    print("=" * 60)

    model, processor = load_grounding_dino()

    # 输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    class_map = {name: i for i, name in enumerate(SELECTED_CLASSES)}

    with open(OUTPUT_DIR / "classes.txt", 'w', encoding='utf-8') as f:
        for name in SELECTED_CLASSES:
            f.write(f"{name}\n")

    total_annotated = 0
    total_boxes = 0
    results = {}

    for disease_class in SELECTED_CLASSES:
        class_id = class_map[disease_class]
        train_dir = ARCHIVE_DIR / "train" / disease_class

        if not train_dir.exists():
            print(f"  [SKIP] {disease_class}: directory not found")
            continue

        images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.JPG"))
        if not images:
            continue

        sample_images = images[:SAMPLES_PER_CLASS]
        prompt = DISEASE_PROMPTS[disease_class]
        disease_cn = CLASS_NAMES_CN.get(disease_class, disease_class)

        print(f"\n--- {disease_cn}({disease_class}): {len(sample_images)}张 ---")
        print(f"  Prompt: \"{prompt}\"")

        class_boxes = 0
        class_annotated = 0

        for i, img_path in enumerate(sample_images):
            try:
                img = Image.open(img_path)
                orig_w, orig_h = img.size

                boxes, scores, labels = detect_disease(model, processor, img_path, prompt)

                if len(boxes) > 0:
                    # 转换并保存
                    yolo_boxes = boxes_to_yolo(boxes, orig_w, orig_h)

                    import shutil
                    out_img_name = f"{disease_class}_{img_path.stem}.jpg"
                    out_img_path = OUTPUT_DIR / "images" / out_img_name
                    out_label_path = OUTPUT_DIR / "labels" / f"{disease_class}_{img_path.stem}.txt"

                    shutil.copy2(img_path, out_img_path)
                    save_yolo_label(out_label_path, yolo_boxes, class_id)

                    class_boxes += len(yolo_boxes)
                    class_annotated += 1

                    # 打印每个框的置信度
                    score_str = ", ".join([f"{s:.2f}" for s in scores[:5]])
                    print(f"  [{i+1}/{len(sample_images)}] {img_path.name}: {len(boxes)} boxes (scores: {score_str})")
                else:
                    print(f"  [{i+1}/{len(sample_images)}] {img_path.name}: no detection (> threshold)")

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")
                import traceback
                traceback.print_exc()

        results[disease_class] = {"images": class_annotated, "boxes": class_boxes}
        total_annotated += class_annotated
        total_boxes += class_boxes

    print(f"\n{'='*60}")
    print(f"标注完成!")
    print(f"{'='*60}")
    for cls, info in results.items():
        print(f"  {cls}: {info['images']}张图, {info['boxes']}个框")
    print(f"\n总计: {total_annotated}张图, {total_boxes}个框")
    print(f"输出: {OUTPUT_DIR}")

    # data.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: images
val: images
nc: {len(SELECTED_CLASSES)}
names: {SELECTED_CLASSES}
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    print("data.yaml 已生成")


if __name__ == "__main__":
    main()
