"""
补充难例挖掘 — 保持原有评分逻辑，补足 to_label_v2 到每类30张(短边>=480)
"""
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
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

TARGET_PER_CLASS = 30
CONF_THRESHOLD = 0.2
MIN_SHORT_SIDE = 480


def main():
    # 读取已选图片名（按类）+ 已标数据集（避免重复）
    existing = {i: set() for i in range(7)}
    for cls_id in range(7):
        cls_dir = OUT_DIR / CLASS_NAMES[cls_id]
        if cls_dir.exists():
            for f in cls_dir.glob("*.jpg"):
                existing[cls_id].add(f.name)

    # 收集所有已标过的图片名（golden_set_7class + to_label）
    already_labeled = set()
    for d in [PROJECT / "golden_set_7class", PROJECT / "to_label"]:
        if d.exists():
            for f in d.rglob("*.jpg"):
                already_labeled.add(f.name)
    print(f"已标图片总数（排除用）: {len(already_labeled)}")

    # 统计当前合格数
    print("=== 当前状态 ===")
    for cls_id in range(7):
        cls_dir = OUT_DIR / CLASS_NAMES[cls_id]
        valid = 0
        if cls_dir.exists():
            for f in cls_dir.glob("*.jpg"):
                try:
                    img = Image.open(f)
                    if min(img.size) >= MIN_SHORT_SIDE:
                        valid += 1
                except:
                    pass
        need = max(0, TARGET_PER_CLASS - valid)
        print(f"  [{cls_id}] {CLASS_NAMES[cls_id]}: 合格{valid}, 已有{len(existing[cls_id])}, 需补{need}")

    print(f"\n{'='*60}")
    print("开始扫描 14k 数据集...")
    print(f"模型: {MODEL_PATH}")
    print(f"短边阈值: {MIN_SHORT_SIDE}px")
    print(f"{'='*60}")

    model = YOLO(str(MODEL_PATH))

    # 收集所有候选 (含未预测到的)
    all_scores = {i: [] for i in range(7)}

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

            # 跳过已选的 + 已标过的
            img_filename = img_path.name
            if img_filename in already_labeled:
                continue
            already_used = False
            for cls_id in range(7):
                if img_filename in existing[cls_id]:
                    already_used = True
                    break
            if already_used:
                continue

            # 检查尺寸
            try:
                img = Image.open(img_path)
                if min(img.size) < MIN_SHORT_SIDE:
                    continue
            except:
                continue

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
                        continue

            if orig_cls_7 is None:
                continue

            if r.boxes is None or len(r.boxes) == 0:
                if orig_cls_7 == 6:  # Healthy: 无预测也有价值
                    all_scores[6].append((5.0, img_path, {"n_preds": 0, "avg_conf": 0, "pred_classes": []}))
                continue

            boxes = r.boxes
            pred_cls = boxes.cls.cpu().numpy().astype(int)
            pred_conf = boxes.conf.cpu().numpy()

            n_preds = len(pred_cls)
            avg_conf = float(pred_conf.mean())
            unique_classes = set(pred_cls.tolist())

            # === 综合评分（与 find_hard_cases.py 完全一致）===
            score = 0.0
            score += min(n_preds, 5)
            if orig_cls_7 not in unique_classes:
                score += 3.0
            if 0.3 <= avg_conf <= 0.6:
                score += 3.0
            elif 0.6 < avg_conf <= 0.8:
                score += 1.5
            score += min(len(unique_classes), 3)

            info = {
                "n_preds": n_preds,
                "avg_conf": round(avg_conf, 3),
                "pred_classes": sorted(list(unique_classes)),
                "orig_class": orig_cls_7,
                "class_match": orig_cls_7 in unique_classes,
            }

            all_scores[orig_cls_7].append((score, img_path, info))

    # ===== 每类补选 =====
    total_added = 0
    for cls_id in range(7):
        cls_dir = OUT_DIR / CLASS_NAMES[cls_id]
        # 统计当前合格数
        valid = 0
        if cls_dir.exists():
            for f in cls_dir.glob("*.jpg"):
                try:
                    img = Image.open(f)
                    if min(img.size) >= MIN_SHORT_SIDE:
                        valid += 1
                except:
                    pass
        need = max(0, TARGET_PER_CLASS - valid)
        if need == 0:
            print(f"[{cls_id}] {CLASS_NAMES[cls_id]}: 已满{valid}张，跳过")
            continue

        items = sorted(all_scores[cls_id], key=lambda x: -x[0])
        # 去重（同一张图可能出现在多个类）
        seen_names = set()
        unique_items = []
        for score, img_path, info in items:
            if img_path.name not in seen_names:
                seen_names.add(img_path.name)
                unique_items.append((score, img_path, info))

        to_add = unique_items[:need]
        cls_dir.mkdir(exist_ok=True)
        for score, img_path, info in to_add:
            shutil.copy2(img_path, cls_dir / img_path.name)
            total_added += 1
            print(f"  + {img_path.name} (score={score:.1f}, n_preds={info['n_preds']}, "
                  f"avg_conf={info['avg_conf']}, classes={info['pred_classes']})")

        print(f"[{cls_id}] {CLASS_NAMES[cls_id]}: 合格{valid} → 补{len(to_add)} → {valid+len(to_add)}")

    print(f"\n{'='*60}")
    print(f"共补充 {total_added} 张")
    print(f"现有图片总数: {sum(1 for _ in OUT_DIR.rglob('*.jpg'))}")
    print("完成！可继续用 label_tool_v2.py 标注")


if __name__ == "__main__":
    main()
