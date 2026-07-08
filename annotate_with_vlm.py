"""
小麦病害 VLM 标注脚本
使用 Qwen2-VL-2B (4-bit) 对小麦病害图像进行 bbox 标注
"""

import os
import json
import re
import time
from pathlib import Path
from PIL import Image
import torch

# ========== 配置 ==========
ARCHIVE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\archive\Wheat_Disease")
OUTPUT_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_detection_vlm")
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# 选几个核心病害类（数据量大的优先）
SELECTED_CLASSES = [
    "Brown Rust",
    "Yellow Rust",
    "Septoria",
    "Mildew",
    "Black Rust",
    "Healthy Wheat",
]

# 每个类先标注这么多张测试
SAMPLES_PER_CLASS = 2  # 测试用，正式改成 50-100

# 图像处理
MAX_IMAGE_SIZE = 1024  # 缩小到1024以内节省显存

# ========== 类别映射 ==========
CLASS_NAMES_CN = {
    "Brown Rust": "小麦褐锈病",
    "Yellow Rust": "小麦黄锈病",
    "Septoria": "小麦壳针孢叶斑病",
    "Mildew": "小麦白粉病",
    "Black Rust": "小麦黑锈病",
    "Healthy Wheat": "健康小麦",
}


def load_model():
    """加载 Qwen2-VL-2B 4-bit 量化模型"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    print(f"Loading {MODEL_NAME} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # 设置更大的 max_new_tokens 以容纳多个 bbox
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB used")
    return model, processor


def resize_image(img, max_size=MAX_IMAGE_SIZE):
    """等比缩放图像"""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def build_prompt(disease_name_en):
    """构建 VLM 标注提示"""
    disease_cn = CLASS_NAMES_CN.get(disease_name_en, disease_name_en)

    if disease_name_en == "Healthy Wheat":
        return f"""This is a healthy wheat leaf image. Are there any visible disease symptoms on this wheat leaf?

If completely healthy, respond with: {{"healthy": true, "boxes": []}}

If you see ANY disease spots or abnormalities, describe their locations with bounding boxes.

For each region, provide normalized coordinates [x_center, y_center, width, height] where values are between 0 and 1 (relative to image width/height).

Respond ONLY in this JSON format:
{{"healthy": false, "boxes": [[x_center, y_center, width, height], ...], "description": "brief description"}}"""

    return f"""This is a wheat leaf image. The plant has {disease_cn} ({disease_name_en}).

Please locate ALL visible disease symptom regions on this leaf. Disease symptoms appear as discolored spots, lesions, rust pustules, or powdery patches.

For each disease region, provide normalized bounding box coordinates [x_center, y_center, width, height] where values are between 0 and 1 (relative to image width/height).

Respond ONLY in this JSON format:
{{"boxes": [[x_center, y_center, width, height], ...], "confidence": "high/medium/low"}}

If no disease is clearly visible, return: {{"boxes": [], "confidence": "none"}}"""


def parse_response(text, img_w, img_h):
    """解析 VLM 返回的 bbox 坐标"""
    # 尝试提取 JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    boxes = data.get("boxes", [])
    if not boxes:
        return []

    # 验证和规范化坐标
    valid_boxes = []
    for box in boxes:
        if len(box) != 4:
            continue
        x, y, w, h = box
        # 确保在 [0,1] 范围内
        if all(0 <= v <= 1.2 for v in [x, y, w, h]):
            # 裁剪到 [0,1]
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            w = max(0.01, min(1, w))
            h = max(0.01, min(1, h))
            valid_boxes.append([x, y, w, h])

    return valid_boxes


def annotate_image(model, processor, image_path, disease_class, class_id):
    """对单张图像进行 VLM 标注"""

    # 加载并缩放图像
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img = resize_image(img)
    new_w, new_h = img.size

    # 缩放因子（VLM 在缩放后的图上给坐标，缩放到原始图）

    prompt = build_prompt(disease_class)

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 处理输入
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 解析 bbox
    boxes = parse_response(response, new_w, new_h)

    return boxes, response


def save_yolo_label(label_path, boxes, class_id):
    """保存 YOLO 格式标注文件"""
    with open(label_path, 'w') as f:
        for box in boxes:
            x, y, w, h = box
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    print("=" * 60)
    print("小麦病害 VLM 标注工具 (Qwen2-VL-2B)")
    print("=" * 60)

    # 加载模型
    model, processor = load_model()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    # 类别映射
    class_map = {name: i for i, name in enumerate(SELECTED_CLASSES)}

    # 保存类别文件
    with open(OUTPUT_DIR / "classes.txt", 'w', encoding='utf-8') as f:
        for name in SELECTED_CLASSES:
            f.write(f"{name}\n")

    total_annotated = 0
    results = {}

    for disease_class in SELECTED_CLASSES:
        class_id = class_map[disease_class]
        train_dir = ARCHIVE_DIR / "train" / disease_class

        if not train_dir.exists():
            print(f"  [SKIP] {disease_class}: directory not found")
            continue

        images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.JPG")) + list(train_dir.glob("*.png"))

        if not images:
            print(f"  [SKIP] {disease_class}: no images found")
            continue

        # 取前 N 张
        sample_images = images[:SAMPLES_PER_CLASS]
        print(f"\n{'='*60}")
        print(f"标注 {disease_class} (class_id={class_id}): {len(sample_images)} 张")
        print(f"{'='*60}")

        class_boxes = 0
        class_annotated = 0

        for img_path in sample_images:
            try:
                boxes, response = annotate_image(model, processor, img_path, disease_class, class_id)

                if boxes:
                    # 复制图像到输出目录
                    import shutil
                    out_img_name = f"{disease_class}_{img_path.stem}.jpg"
                    out_img_path = OUTPUT_DIR / "images" / out_img_name
                    out_label_path = OUTPUT_DIR / "labels" / f"{disease_class}_{img_path.stem}.txt"

                    shutil.copy2(img_path, out_img_path)
                    save_yolo_label(out_label_path, boxes, class_id)

                    class_boxes += len(boxes)
                    class_annotated += 1
                    print(f"  [{class_annotated}/{len(sample_images)}] {img_path.name}: {len(boxes)} boxes")
                else:
                    print(f"  [{class_annotated}/{len(sample_images)}] {img_path.name}: no disease detected")

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

        results[disease_class] = {
            "images": class_annotated,
            "boxes": class_boxes,
        }
        total_annotated += class_annotated

    # 汇总
    print(f"\n{'='*60}")
    print(f"标注完成！")
    print(f"{'='*60}")
    for cls, info in results.items():
        print(f"  {cls}: {info['images']} 张图, {info['boxes']} 个框")
    print(f"\n总计: {total_annotated} 张图")
    print(f"输出目录: {OUTPUT_DIR}")

    # 生成 data.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: images
val: images
nc: {len(SELECTED_CLASSES)}
names: {SELECTED_CLASSES}
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    print(f"data.yaml 已生成")


if __name__ == "__main__":
    main()
