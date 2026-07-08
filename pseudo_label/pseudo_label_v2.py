"""
V2模型 → 14k张图推理 → 第2轮伪标注
策略同V1但降低阈值 (模型更强了)
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL_PATH = PROJECT / "runs" / "wheat_7class_v2" / "weights" / "best.pt"
SRC_DATASET = PROJECT / "wheat_disease_8class"
OUT_DIR = PROJECT / "wheat_pseudo_v2"

MAP_8TO7 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6}
CONF_THRESHOLD = 0.25
CONF_HEALTHY = 0.5

CLASS_NAMES = [
    "Brown Rust", "Yellow Rust", "Black Rust", "Septoria",
    "Powdery Mildew", "Fusarium Head Blight", "Healthy Wheat",
]


def main():
    print("=" * 60)
    print("V2 伪标注流水线")
    print(f"模型: {MODEL_PATH}")
    print(f"置信度: {CONF_THRESHOLD}")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ["train", "val"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"total_imgs": 0, "matched_preds": 0, "total_preds": 0,
             "kept_preds": 0, "leaf_blight_skip": 0,
             "per_class_kept": {i: 0 for i in range(7)}}

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
            stats["total_imgs"] += 1

            # 读原标签类别
            lbl_path = lbl_dir / f"{img_name}.txt"
            if not lbl_path.exists():
                continue
            lines = lbl_path.read_text().strip().splitlines()
            if not lines or not lines[0].strip():
                continue
            orig_cls_8 = int(lines[0].split()[0])
            if orig_cls_8 not in MAP_8TO7:
                stats["leaf_blight_skip"] += 1
                continue
            target_cls_7 = MAP_8TO7[orig_cls_8]

            if r.boxes is None or len(r.boxes) == 0:
                if target_cls_7 == 6:
                    shutil.copy2(img_path, OUT_DIR / split / "images" / img_path.name)
                    (OUT_DIR / split / "labels" / f"{img_name}.txt").write_text("")
                continue

            boxes = r.boxes
            pred_cls = boxes.cls.cpu().numpy().astype(int)
            pred_conf = boxes.conf.cpu().numpy()
            pred_xywhn = boxes.xywhn.cpu().numpy()

            thresh = CONF_HEALTHY if target_cls_7 == 6 else CONF_THRESHOLD
            kept = []
            for i in range(len(pred_cls)):
                stats["total_preds"] += 1
                if pred_cls[i] == target_cls_7 and pred_conf[i] >= thresh:
                    kept.append(i)

            if kept:
                stats["matched_preds"] += 1
                for i in kept:
                    stats["kept_preds"] += 1
                    stats["per_class_kept"][pred_cls[i]] = stats["per_class_kept"].get(pred_cls[i], 0) + 1
                shutil.copy2(img_path, OUT_DIR / split / "images" / img_path.name)
                with open(OUT_DIR / split / "labels" / f"{img_name}.txt", "w") as f:
                    for i in kept:
                        cx, cy, w, h = pred_xywhn[i]
                        f.write(f"{pred_cls[i]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"\n{'='*60}")
    print(f"V2 伪标注完成")
    print(f"{'='*60}")
    print(f"处理图片: {stats['total_imgs']}")
    print(f"有伪标注: {stats['matched_preds']} (+{(stats['matched_preds']/stats['total_imgs']*100):.1f}%)")
    print(f"保留框: {stats['kept_preds']}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}: {stats['per_class_kept'].get(i, 0)}")

    for split in ["train", "val"]:
        imgs = len(list((OUT_DIR / split / "images").glob("*")))
        lbls = len(list((OUT_DIR / split / "labels").glob("*")))
        print(f"\n{split}: {imgs} images, {lbls} labels")

    (OUT_DIR / "data.yaml").write_text(f"""path: {OUT_DIR}
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
""", encoding="utf-8")
    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
