"""
用V3模型扫描14k数据集，找最难/最不确定的图片供手工标注
策略：多框、低置信度、类别不匹配 = 高价值候选
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import json

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL_PATH = PROJECT / "runs" / "wheat_7class_v33" / "weights" / "best.pt"
SRC_DATASET = PROJECT / "wheat_disease_8class"
OUT_DIR = PROJECT / "to_label_v2"

MAP_8TO7 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6}
CLASS_NAMES = [
    "Brown_Rust", "Yellow_Rust", "Black_Rust", "Septoria",
    "Powdery_Mildew", "Fusarium_Head_Blight", "Healthy_Wheat",
]

# 每类挑多少张
PER_CLASS = 30
CONF_THRESHOLD = 0.2  # 低阈值，收集更多预测


def main():
    print("=" * 60)
    print("V3 难例挖掘")
    print(f"模型: {MODEL_PATH}")
    print(f"每类选取: {PER_CLASS} 张")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    # 收集所有图片的预测结果
    all_scores = {i: [] for i in range(7)}  # class -> [(score, img_path, info)]

    for split in ["train", "val"]:
        img_dir = SRC_DATASET / split / "images"
        lbl_dir = SRC_DATASET / split / "labels"
        if not img_dir.exists():
            continue

        results = model.predict(
            source=str(img_dir), conf=CONF_THRESHOLD, iou=0.5,
            imgsz=640, device=0, stream=True, verbose=False,
        )

        for r in results:
            img_path = Path(r.path)
            img_name = img_path.stem

            # 读原标签
            lbl_path = lbl_dir / f"{img_name}.txt"
            orig_cls_7 = None
            if lbl_path.exists():
                lines = lbl_path.read_text().strip().splitlines()
                if lines and lines[0].strip():
                    orig_cls_8 = int(lines[0].split()[0])
                    if orig_cls_8 in MAP_8TO7:
                        orig_cls_7 = MAP_8TO7[orig_cls_8]
                    else:
                        continue  # Leaf Blight skip

            if orig_cls_7 is None:
                continue

            if r.boxes is None or len(r.boxes) == 0:
                # 无预测 → 可能漏检，对Healthy有价值
                if orig_cls_7 == 6:
                    all_scores[6].append((5.0, img_path, {"n_preds": 0, "avg_conf": 0, "pred_classes": []}))
                continue

            boxes = r.boxes
            pred_cls = boxes.cls.cpu().numpy().astype(int)
            pred_conf = boxes.conf.cpu().numpy()

            n_preds = len(pred_cls)
            avg_conf = float(pred_conf.mean())
            unique_classes = set(pred_cls.tolist())

            # === 综合评分（越高越值得标）===
            score = 0.0

            # 1. 多框加分（最多5分）
            score += min(n_preds, 5)

            # 2. 类别不匹配加分（最多3分）
            if orig_cls_7 not in unique_classes:
                score += 3.0  # 模型看到的跟标签完全不一样

            # 3. 中等置信度加分（0.3-0.6区间最不确定，最多3分）
            if 0.3 <= avg_conf <= 0.6:
                score += 3.0
            elif 0.6 < avg_conf <= 0.8:
                score += 1.5

            # 4. 多类预测加分（最多3分）
            score += min(len(unique_classes), 3)

            info = {
                "n_preds": n_preds,
                "avg_conf": round(avg_conf, 3),
                "pred_classes": sorted(list(unique_classes)),
                "orig_class": orig_cls_7,
                "class_match": orig_cls_7 in unique_classes,
            }

            all_scores[orig_cls_7].append((score, img_path, info))

    # ===== 每类选TOP-N =====
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    selected = []
    for cls_id in range(7):
        items = sorted(all_scores[cls_id], key=lambda x: -x[0])
        top_n = items[:PER_CLASS]
        cls_dir = OUT_DIR / CLASS_NAMES[cls_id]
        cls_dir.mkdir()
        for score, img_path, info in top_n:
            shutil.copy2(img_path, cls_dir / img_path.name)
            selected.append({
                "class": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "image": img_path.name,
                "score": round(score, 1),
                **info,
            })
        print(f"[{cls_id}] {CLASS_NAMES[cls_id]}: {len(items)} available → 选 {len(top_n)}")

    # 保存元信息
    meta = {
        "total_selected": len(selected),
        "per_class": PER_CLASS,
        "model": str(MODEL_PATH),
        "images": selected,
    }
    (OUT_DIR / "selection_info.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"选中 {len(selected)} 张 → {OUT_DIR}")
    print("每类一个子文件夹，直接用 label_tool_v2.py 标注即可")


if __name__ == "__main__":
    main()
