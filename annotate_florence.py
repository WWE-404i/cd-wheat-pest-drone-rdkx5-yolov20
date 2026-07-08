"""
Florence-2-large 自动标注脚本
- 模型定位病变区域, 文件夹名给类别
- 只标注 image_audit/keep 中的图片 (>=640px)
- Healthy Wheat 跳过, 给空标签
- 输出 YOLO 格式到 wheat_yolo_train_f2/
- 支持断点续跑
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
import random

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
ARCHIVE = PROJECT / "archive" / "Wheat_Disease" / "train"
KEEP_DIR = PROJECT / "image_audit" / "keep_短边大于等于640"
OUTPUT_DIR = PROJECT / "wheat_yolo_train_f2"
CHECKPOINT = PROJECT / "florence_checkpoint.json"

# 4 类映射：文件夹名 -> class_id
CLASS_MAP = {
    "Brown Rust": 0,
    "Black Rust": 1,
    "Septoria": 2,
    "Healthy Wheat": 3,
}
CLASS_NAMES = list(CLASS_MAP.keys())

# 配置
CONFIDENCE_THRESHOLD = 0.3
TRAIN_RATIO = 0.85
BATCH_SIZE = 1  # Florence-2 推理

# --- 模型加载 ---
print(f"[{datetime.now():%H:%M:%S}] Loading microsoft/Florence-2-large...")
from transformers import AutoProcessor, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True,
)

vram = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
print(f"[{datetime.now():%H:%M:%S}] Model ready. Device={DEVICE}, VRAM={vram:.1f}GB")


# --- 收集待标注图片 ---
def collect_images():
    """扫描 keep 目录, 按文件夹分类, 返回 (图片路径, class_id, 文件夹名) 列表"""
    items = []
    for cls_folder, cls_id in CLASS_MAP.items():
        src_dir = ARCHIVE / cls_folder
        if not src_dir.exists():
            print(f"  SKIP: {cls_folder} not found in archive")
            continue

        # 获取 keep 列表中的文件名
        keep_list_file = KEEP_DIR / f"{cls_folder}.txt"
        if keep_list_file.exists():
            # 如果有 keep 列表文件, 按列表来
            with open(keep_list_file) as f:
                keep_names = set(line.strip() for line in f if line.strip())
            for img_file in src_dir.iterdir():
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    if img_file.stem in keep_names:
                        items.append((img_file, cls_id, cls_folder))
        else:
            # 没有 keep 列表, 直接用所有 >=640px 的图
            # 简化: 直接从 keep 目录对应的结构读取
            pass

    print(f"  收集到 {len(items)} 张待标注图片")
    return items


REVIEW_DIR = PROJECT / "image_audit" / "manual_review_480到639"


def scan_dir(base_dir):
    """扫描一个目录, 返回 {cls_name: [img_path, ...]}"""
    result = {}
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        cls_name = subdir.name
        if cls_name not in CLASS_MAP:
            continue
        imgs = [f for f in subdir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if imgs:
            result[cls_name] = imgs
    return result


def collect_images_v2():
    """扫描 keep + review 两个目录"""
    items = []

    for source_name, source_dir in [("keep", KEEP_DIR), ("review", REVIEW_DIR)]:
        if not source_dir.exists():
            print(f"  {source_name}: 目录不存在, 跳过")
            continue
        scanned = scan_dir(source_dir)
        total = sum(len(v) for v in scanned.values())
        print(f"  {source_name}: {total} 张")
        for cls_name, imgs in scanned.items():
            cls_id = CLASS_MAP[cls_name]
            for img_file in imgs:
                items.append((img_file, cls_id, cls_name))

    print(f"  合计: {len(items)} 张待标注图片")
    return items


# --- Florence-2 推理 ---
def detect(img_path):
    """Florence-2 目标检测, 返回 [{x1,y1,x2,y2,confidence}, ...]"""
    from PIL import Image
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return []

    w, h = image.size

    # 使用 OPEN_VOCABULARY_DETECTION: 告诉 Florence-2 找什么
    prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_prompt = "disease spot. lesion. rust. blotch. mildew spot."
    try:
        inputs = processor(text=prompt + text_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    except Exception:
        return []

    with torch.no_grad():
        try:
            # convert float inputs to match model dtype (fp16)
            inputs = {k: v.to(DEVICE, dtype=model.dtype) if v.dtype in (torch.float32, torch.float16)
                      else v.to(DEVICE)
                      for k, v in inputs.items()}
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        except Exception as e:
            print(f"    generate error: {e}")
            return []

    try:
        results = processor.post_process_generation(
            generated_ids,
            task="<OPEN_VOCABULARY_DETECTION>",
            image_size=(w, h),
        )
    except Exception:
        return []

    # 解析结果
    boxes = []
    parsed = results[0] if isinstance(results, list) else results
    if isinstance(parsed, dict):
        od_results = parsed.get("<OPEN_VOCABULARY_DETECTION>", [])
        for det in od_results:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                conf = det.get("score", 0.5) if "score" in det else 0.5
                if conf >= CONFIDENCE_THRESHOLD:
                    boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": conf,
                    })

    return boxes


def boxes_to_yolo(boxes, img_w, img_h):
    """转为 YOLO 归一化格式 [class_id, cx, cy, w, h]"""
    yolo_lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        bw = x2 - x1
        bh = y2 - y1
        # 过滤太小/太大的框
        if bw < 5 or bh < 5:
            continue
        if bw > img_w * 0.95 or bh > img_h * 0.95:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        nw = bw / img_w
        nh = bh / img_h
        yolo_lines.append((cx, cy, nw, nh))
    return yolo_lines


# --- 主流程 ---
def main():
    # 检查 checkpoint
    processed = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        processed = set(ckpt.get("processed", []))
        print(f"[{datetime.now():%H:%M:%S}] Resume from checkpoint: {len(processed)} done")

    # 收集图片
    items = collect_images_v2()
    if not items:
        print("No images to annotate! Check image_audit/keep directory.")
        return

    # 过滤已处理的
    remaining = [(img, cid, cname) for img, cid, cname in items
                 if img.stem not in processed]
    print(f"[{datetime.now():%H:%M:%S}] To annotate: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    # 创建输出目录
    (OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # 打乱顺序
    random.shuffle(remaining)

    t0 = time.time()
    ok_count = 0
    box_total = 0
    empty_count = 0

    for idx, (img_path, cls_id, cls_name) in enumerate(remaining):
        # 决定 train/val
        is_train = random.random() < TRAIN_RATIO
        split = "train" if is_train else "val"

        # Healthy Wheat: 跳过推理, 空标签
        if cls_name == "Healthy Wheat":
            boxes = []
        else:
            boxes = detect(img_path)
            box_total += len(boxes)

        # 转 YOLO 格式
        from PIL import Image
        try:
            img_obj = Image.open(img_path)
            img_w, img_h = img_obj.size
        except Exception:
            print(f"  SKIP: cannot read {img_path.name}")
            continue

        yolo_boxes = boxes_to_yolo(boxes, img_w, img_h)

        # 写 label
        label_name = f"{cls_name}_{img_path.stem}.txt"
        label_path = OUTPUT_DIR / split / "labels" / label_name
        with open(label_path, "w", encoding="utf-8") as f:
            for cx, cy, nw, nh in yolo_boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        # 复制/链接图片 (直接复制, 保持简单)
        import shutil
        img_dst = OUTPUT_DIR / split / "images" / f"{cls_name}_{img_path.stem}{img_path.suffix}"
        if not img_dst.exists():
            shutil.copy2(img_path, img_dst)

        ok_count += 1
        if not yolo_boxes:
            empty_count += 1

        # 进度
        if (idx + 1) % 50 == 0 or idx < 5:
            elapsed = time.time() - t0
            spd = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - idx - 1) / spd if spd > 0 else 0
            print(f"[{datetime.now():%H:%M:%S}]   [{idx+1}/{len(remaining)}] "
                  f"ok={ok_count} boxes_avg={box_total/max(ok_count-empty_count,1):.1f} "
                  f"spd={spd:.1f}/s ETA={eta/60:.0f}min")

        # 每 100 张存 checkpoint
        if (idx + 1) % 100 == 0:
            processed.add(img_path.stem)
            with open(CHECKPOINT, "w") as f:
                json.dump({"processed": list(processed), "last_idx": idx + 1}, f)

    # 最终保存
    processed.update(img.stem for img, _, _ in remaining)
    with open(CHECKPOINT, "w") as f:
        json.dump({"processed": list(processed), "done": True}, f)

    # 写 data.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images
nc: 7
names: {CLASS_NAMES}
"""
    with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    elapsed = time.time() - t0
    print(f"\n[{datetime.now():%H:%M:%S}] Done! {ok_count} images, "
          f"{box_total} boxes, {elapsed/60:.0f}min")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  data.yaml ready for training")


if __name__ == "__main__":
    main()
