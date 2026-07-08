"""
GD Tiling 速度基准测试 —— 对比不同优化策略
策略:
  A) 逐 tile 逐 prompt (当前 baseline)
  B) 逐 tile 合并 prompt (prompt concat)
  C) Batch tiles + 合并 prompt
  D) Batch tiles + 合并 prompt + FP16
  E) Batch tiles + 合并 prompt + FP16 + torch.compile (RTX 40xx+)
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import torch
import numpy as np
import cv2
import time
from pathlib import Path
from PIL import Image
from datetime import datetime

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
KEEP_DIR = PROJECT / "image_audit" / "keep_短边大于等于640"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TILE_SIZE = 640
TILE_STRIDE = 512

PROMPTS = {
    "Brown Rust": [
        "wheat leaf scattered with small oval brown rust pustules",
        "discrete tiny brown lesion spots on leaf surface",
    ],
    "Black Rust": [
        "wheat leaf with large dark reddish-black clustered rust pustules",
        "irregular big dark rust patches on wheat blade",
    ],
    "Septoria": [
        "wheat leaf with small dark brown angular necrotic speck lesions",
        "scattered dark spot blotch on wheat leaf surface",
    ],
}

# ============================================
# 加载模型
# ============================================
print(f"[{datetime.now():%H:%M:%S}] Loading Grounding DINO-tiny...")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id, local_files_only=True
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

vram = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
print(f"  GPU: {gpu_name}")
print(f"  VRAM used: {vram:.1f} GB")
print(f"  FP16 support: {torch.cuda.is_bf16_supported() if DEVICE=='cuda' else 'N/A'}")


# ============================================
# 策略 A: Baseline (逐 tile 逐 prompt)
# ============================================
def strategy_A(img_bgr, prompts):
    """当前 annotate_gd_tile.py 的做法"""
    img_h, img_w = img_bgr.shape[:2]

    tiles = []
    for y in range(0, img_h, TILE_STRIDE):
        for x in range(0, img_w, TILE_STRIDE):
            tx2 = min(x + TILE_SIZE, img_w)
            ty2 = min(y + TILE_SIZE, img_h)
            tx1 = tx2 - TILE_SIZE if tx2 >= img_w else x
            ty1 = ty2 - TILE_SIZE if ty2 >= img_h else y
            tiles.append((tx1, ty1, tx2, ty2))
    tiles = list(set(tiles))  # dedup

    all_boxes = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile_bgr = img_bgr[ty1:ty2, tx1:tx2]
        img_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        for prompt in prompts:
            with torch.no_grad():
                inputs = processor(images=image, text=prompt, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"],
                    box_threshold=0.15, text_threshold=0.10,
                    target_sizes=[image.size[::-1]],
                )
            for det in results[0]["boxes"]:
                box = det.tolist()
                all_boxes.append([
                    box[0] + tx1, box[1] + ty1,
                    box[2] + tx1, box[3] + ty1
                ])

    return all_boxes


# ============================================
# 策略 B: 逐 tile + 合并 prompt
# ============================================
def strategy_B(img_bgr, prompts):
    """每个 tile 一次 GD 推理，用合并的 prompt"""
    img_h, img_w = img_bgr.shape[:2]
    combined_prompt = " . ".join(prompts)

    tiles = []
    for y in range(0, img_h, TILE_STRIDE):
        for x in range(0, img_w, TILE_STRIDE):
            tx2 = min(x + TILE_SIZE, img_w)
            ty2 = min(y + TILE_SIZE, img_h)
            tx1 = tx2 - TILE_SIZE if tx2 >= img_w else x
            ty1 = ty2 - TILE_SIZE if ty2 >= img_h else y
            tiles.append((tx1, ty1, tx2, ty2))
    tiles = list(set(tiles))

    all_boxes = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile_bgr = img_bgr[ty1:ty2, tx1:tx2]
        img_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        with torch.no_grad():
            inputs = processor(images=image, text=combined_prompt, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                box_threshold=0.15, text_threshold=0.10,
                target_sizes=[image.size[::-1]],
            )
        for det in results[0]["boxes"]:
            box = det.tolist()
            all_boxes.append([
                box[0] + tx1, box[1] + ty1,
                box[2] + tx1, box[3] + ty1
            ])

    return all_boxes


# ============================================
# 策略 C: Batch tiles + 合并 prompt
# ============================================
def strategy_C(img_bgr, prompts):
    """所有 tiles 打包成一个 batch 推理"""
    img_h, img_w = img_bgr.shape[:2]
    combined_prompt = " . ".join(prompts)

    tiles = []
    for y in range(0, img_h, TILE_STRIDE):
        for x in range(0, img_w, TILE_STRIDE):
            tx2 = min(x + TILE_SIZE, img_w)
            ty2 = min(y + TILE_SIZE, img_h)
            tx1 = tx2 - TILE_SIZE if tx2 >= img_w else x
            ty1 = ty2 - TILE_SIZE if ty2 >= img_h else y
            tiles.append((tx1, ty1, tx2, ty2))
    tiles = list(set(tiles))

    # Batch 所有 tiles
    pil_images = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile_bgr = img_bgr[ty1:ty2, tx1:tx2]
        img_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(img_rgb))

    all_boxes = []
    BATCH_SIZE = 8  # 防止 OOM

    for i in range(0, len(pil_images), BATCH_SIZE):
        batch_imgs = pil_images[i:i+BATCH_SIZE]
        batch_texts = [combined_prompt] * len(batch_imgs)

        with torch.no_grad():
            inputs = processor(images=batch_imgs, text=batch_texts, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            target_sizes = [(img.size[1], img.size[0]) for img in batch_imgs]
            results = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                box_threshold=0.15, text_threshold=0.10,
                target_sizes=target_sizes,
            )

        for j, (tx1, ty1, tx2, ty2) in enumerate(tiles[i:i+BATCH_SIZE]):
            for det in results[j]["boxes"]:
                box = det.tolist()
                all_boxes.append([
                    box[0] + tx1, box[1] + ty1,
                    box[2] + tx1, box[3] + ty1
                ])

    return all_boxes


# ============================================
# 策略 D: Batch tiles + 合并 prompt + FP16
# ============================================
def strategy_D(img_bgr, prompts):
    """Batch + FP16 autocast"""
    img_h, img_w = img_bgr.shape[:2]
    combined_prompt = " . ".join(prompts)

    tiles = []
    for y in range(0, img_h, TILE_STRIDE):
        for x in range(0, img_w, TILE_STRIDE):
            tx2 = min(x + TILE_SIZE, img_w)
            ty2 = min(y + TILE_SIZE, img_h)
            tx1 = tx2 - TILE_SIZE if tx2 >= img_w else x
            ty1 = ty2 - TILE_SIZE if ty2 >= img_h else y
            tiles.append((tx1, ty1, tx2, ty2))
    tiles = list(set(tiles))

    pil_images = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile_bgr = img_bgr[ty1:ty2, tx1:tx2]
        img_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(img_rgb))

    all_boxes = []
    BATCH_SIZE = 8

    for i in range(0, len(pil_images), BATCH_SIZE):
        batch_imgs = pil_images[i:i+BATCH_SIZE]
        batch_texts = [combined_prompt] * len(batch_imgs)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(DEVICE=="cuda")):
                inputs = processor(images=batch_imgs, text=batch_texts, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs)
            target_sizes = [(img.size[1], img.size[0]) for img in batch_imgs]
            results = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                box_threshold=0.15, text_threshold=0.10,
                target_sizes=target_sizes,
            )

        for j, (tx1, ty1, tx2, ty2) in enumerate(tiles[i:i+BATCH_SIZE]):
            for det in results[j]["boxes"]:
                box = det.tolist()
                all_boxes.append([
                    box[0] + tx1, box[1] + ty1,
                    box[2] + tx1, box[3] + ty1
                ])

    return all_boxes


# ============================================
# 策略 E: Batch + prompt + FP16 + torch.compile (需 Triton, Windows不支持)
# ============================================
strategy_E = None  # torch.compile needs Triton, not available on Windows
# _COMPILE_OK = False
# if DEVICE == "cuda":
#     try:
#         compiled_model = torch.compile(model, mode="reduce-overhead")
#         _COMPILE_OK = True
#         print("  torch.compile: OK")
#     except Exception as e:
#         print(f"  torch.compile: SKIP ({e})")
# Strategy E disabled - torch.compile needs Triton (not available on Windows)


# ============================================
# 选测试图片（不同分辨率）
# ============================================
def find_test_images():
    """找 3 张不同分辨率的测试图"""
    test_cases = []
    for cls_name in ["Black Rust", "Brown Rust", "Septoria"]:
        cls_dir = KEEP_DIR / cls_name
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                with open(f, 'rb') as fh:
                    data = np.frombuffer(fh.read(), np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    test_cases.append((f, w, h, cls_name))
                    break

    # 按分辨率排序
    test_cases.sort(key=lambda x: x[1] * x[2])
    # 取小、中、大各一张
    selected = []
    if test_cases:
        selected.append(test_cases[len(test_cases)//4])  # 小
        selected.append(test_cases[len(test_cases)//2])  # 中
        if len(test_cases) > 2:
            selected.append(test_cases[-1])  # 大
    return selected


# ============================================
# 主测试
# ============================================
def main():
    test_images = find_test_images()
    print(f"\n{'='*70}")
    print(f"Benchmark: {len(test_images)} 张测试图, TILE={TILE_SIZE}, STRIDE={TILE_STRIDE}")
    print(f"{'='*70}")

    for img_path, w, h, cls_name in test_images:
        prompts = PROMPTS[cls_name]
        with open(img_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # 计算 tile 数量
        ntiles = 0
        seen = set()
        for y in range(0, h, TILE_STRIDE):
            for x in range(0, w, TILE_STRIDE):
                tx2 = min(x + TILE_SIZE, w)
                ty2 = min(y + TILE_SIZE, h)
                tx1 = tx2 - TILE_SIZE if tx2 >= w else x
                ty1 = ty2 - TILE_SIZE if ty2 >= h else y
                seen.add((tx1, ty1, tx2, ty2))
        ntiles = len(seen)

        print(f"\n--- {cls_name}: {w}x{h}, {ntiles} tiles, {len(prompts)} prompts ---")

        strategies = {
            "A) tile×prompt (baseline)": (strategy_A, 1),
            "B) tile+merged_prompt": (strategy_B, 1),
            "C) batch_tile+merged": (strategy_C, 2),
            "D) batch+merged+FP16": (strategy_D, 2),
        }
        if strategy_E:
            strategies["E) batch+FP16+compile"] = (strategy_E, 3)

        results = {}
        for name, (func, warmup_runs) in strategies.items():
            # Warmup
            for _ in range(warmup_runs):
                _ = func(img_bgr, prompts)
            torch.cuda.synchronize()

            # Timed runs
            t0 = time.perf_counter()
            boxes = func(img_bgr, prompts)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            results[name] = (dt, len(boxes))

            n_forward = ntiles * len(prompts) if "A" in name else ntiles
            print(f"  {name:<32}: {dt:.2f}s, {len(boxes):>3} boxes, ~{n_forward} forward passes")

        # Speedup vs baseline
        base_time = results[list(strategies.keys())[0]][0]
        print(f"  {'Speedup vs baseline':<32}: ", end="")
        speeds = []
        for name in strategies:
            if name in results:
                sp = base_time / max(results[name][0], 0.001)
                speeds.append(f"{sp:.1f}x")
        print(", ".join(speeds))

    # GPU 信息总结
    print(f"\n{'='*70}")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    if DEVICE == "cuda":
        print(f"Max VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f} GB")
    print(f"Best strategy for batch annotation: D (batch+merged+FP16)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
