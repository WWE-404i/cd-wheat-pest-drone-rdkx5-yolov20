"""
V1模型 → 14k张图推理 → 伪标注筛选
策略: 模型预测类别 = 原标签类别 + 置信度≥阈值 → 保留为新标签
      抛弃全图框，用模型预测框替代
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL_PATH = PROJECT / "runs" / "wheat_7class_manual_v13" / "weights" / "best.pt"
SRC_DATASET = PROJECT / "wheat_disease_8class"
OUT_DIR = PROJECT / "wheat_pseudo_v1"

# 8类→7类映射 (6=Leaf Blight 丢弃)
MAP_8TO7 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6}

CONF_THRESHOLD = 0.3      # 普通类置信度阈值
CONF_HEALTHY = 0.5        # Healthy Wheat 更高阈值（原标签不可靠）

CLASS_NAMES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Powdery Mildew", "Fusarium Head Blight", "Healthy Wheat",
]


def main():
    print("=" * 60)
    print("V1 伪标注流水线")
    print(f"模型: {MODEL_PATH}")
    print(f"置信度阈值: 普通={CONF_THRESHOLD}, Healthy={CONF_HEALTHY}")
    print("=" * 60)

    # 加载模型
    model = YOLO(str(MODEL_PATH))

    # 清空输出
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ["train", "val"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"total_imgs": 0, "has_preds": 0, "matched_preds": 0, "total_preds": 0, "kept_preds": 0,
             "leaf_blight_skip": 0, "class_mismatch": 0, "per_class_kept": {i: 0 for i in range(7)},
             "per_class_total": {i: 0 for i in range(7)}}

    for split in ["train", "val"]:
        img_dir = SRC_DATASET / split / "images"
        lbl_dir = SRC_DATASET / split / "labels"
        if not img_dir.exists():
            continue

        # 收集所有图片
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.png"))
        print(f"\n{split}: {len(images)} 张图")

        # 批量推理
        results = model.predict(
            source=str(img_dir),
            conf=CONF_THRESHOLD,
            iou=0.5,
            imgsz=640,
            device=0,
            stream=True,
            verbose=False,
        )

        for r in results:
            img_path = Path(r.path)
            img_name = img_path.stem
            stats["total_imgs"] += 1

            # 读原标签获取"正确"类别 (8类ID)
            lbl_path = lbl_dir / f"{img_name}.txt"
            orig_cls_8 = None
            if lbl_path.exists():
                lines = lbl_path.read_text().strip().splitlines()
                if lines and lines[0].strip():
                    orig_cls_8 = int(lines[0].split()[0])
                    if orig_cls_8 in MAP_8TO7:
                        target_cls_7 = MAP_8TO7[orig_cls_8]
                    else:
                        # Leaf Blight → 跳过
                        stats["leaf_blight_skip"] += 1
                        continue
                else:
                    # 空标签 → 跳过
                    continue
            else:
                continue

            # 获取模型预测
            if r.boxes is None or len(r.boxes) == 0:
                # 无预测: Healthy Wheat 保留空标签，其他类跳过
                if target_cls_7 == 6:  # Healthy
                    shutil.copy2(img_path, OUT_DIR / split / "images" / img_path.name)
                    (OUT_DIR / split / "labels" / f"{img_name}.txt").write_text("")
                    stats["has_preds"] += 1
                continue

            # 解析预测
            boxes = r.boxes
            pred_cls = boxes.cls.cpu().numpy().astype(int)
            pred_conf = boxes.conf.cpu().numpy()
            pred_xywhn = boxes.xywhn.cpu().numpy()

            has_kept = False
            for i in range(len(pred_cls)):
                stats["total_preds"] += 1
                stats["per_class_total"][pred_cls[i]] = stats["per_class_total"].get(pred_cls[i], 0) + 1

                # 检查类别是否匹配
                if pred_cls[i] != target_cls_7:
                    stats["class_mismatch"] += 1
                    continue

                # Healthy Wheat 用更高阈值
                thresh = CONF_HEALTHY if target_cls_7 == 6 else CONF_THRESHOLD
                if pred_conf[i] < thresh:
                    continue

                # 保留此框
                stats["kept_preds"] += 1
                stats["per_class_kept"][pred_cls[i]] = stats["per_class_kept"].get(pred_cls[i], 0) + 1
                has_kept = True

            if has_kept:
                stats["has_preds"] += 1
                stats["matched_preds"] += 1
                # 复制图片 + 写标签
                shutil.copy2(img_path, OUT_DIR / split / "images" / img_path.name)
                lbl_out = OUT_DIR / split / "labels" / f"{img_name}.txt"
                with open(lbl_out, "w") as f:
                    for i in range(len(pred_cls)):
                        if pred_cls[i] != target_cls_7:
                            continue
                        thresh = CONF_HEALTHY if target_cls_7 == 6 else CONF_THRESHOLD
                        if pred_conf[i] < thresh:
                            continue
                        cx, cy, w, h = pred_xywhn[i]
                        f.write(f"{pred_cls[i]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # 统计
    print(f"\n{'='*60}")
    print(f"伪标注完成")
    print(f"{'='*60}")
    print(f"处理图片: {stats['total_imgs']}")
    print(f"Leaf Blight 跳过: {stats['leaf_blight_skip']}")
    print(f"有伪标注的图: {stats['matched_preds']}")
    print(f"总预测框: {stats['total_preds']}")
    print(f"类不匹配丢弃: {stats['class_mismatch']}")
    print(f"保留框: {stats['kept_preds']}")
    print(f"\n各类保留框:")
    for i, name in enumerate(CLASS_NAMES):
        k = stats["per_class_kept"].get(i, 0)
        t = stats["per_class_total"].get(i, 0)
        print(f"  [{i}] {name}: 保留 {k} / 预测 {t}")

    # 输出数据集大小
    for split in ["train", "val"]:
        imgs = len(list((OUT_DIR / split / "images").glob("*")))
        lbls = len(list((OUT_DIR / split / "labels").glob("*")))
        print(f"\n{split}: {imgs} images, {lbls} labels")

    # 写 data.yaml
    yaml_content = f"""path: {OUT_DIR}
train: train/images
val: val/images
nc: 7
names:
  0: Brown Rust
  1: Yellow Rust
  2: Black Rust
  3: Septoria
  4: Powdery Mildew
  5: Fusarium Head Blight
  6: Healthy Wheat
"""
    (OUT_DIR / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\ndata.yaml → {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
