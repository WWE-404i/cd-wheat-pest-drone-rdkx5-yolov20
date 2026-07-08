"""
V5伪标注 — 用V5 best.pt扫14k，阈值同V3但模型更强
预期: 弱类(Black_Rust/Septoria)检出率自然提升
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL_PATH = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"
SRC_DATASET = PROJECT / "wheat_disease_8class"
OUT_DIR = PROJECT / "wheat_pseudo_v5"
GOLDEN_DIR = PROJECT / "wheat_disease_golden"
OLD_PSEUDO = PROJECT / "wheat_pseudo_v3"

MAP_8TO7 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6}
CONF_THRESHOLD = 0.35
CONF_HEALTHY = 0.5

CLASS_NAMES = [
    "Brown_Rust", "Yellow_Rust", "Black_Rust", "Septoria",
    "Powdery_Mildew", "Fusarium_Head_Blight", "Healthy_Wheat",
]


def main():
    print("=" * 60)
    print("V5 伪标注流水线")
    print(f"模型: {MODEL_PATH}")
    print(f"置信度: 普通={CONF_THRESHOLD}, Healthy={CONF_HEALTHY}")
    print("=" * 60)

    # 只排除精标（伪标注全部重做，V5替换V3）
    exclude = set()
    if GOLDEN_DIR.exists():
        for f in GOLDEN_DIR.rglob("*.jpg"):
            exclude.add(f.stem)
    print(f"排除(精标): {len(exclude)} 张")

    model = YOLO(str(MODEL_PATH))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ["train", "val"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0, "skipped": 0, "leaf_blight": 0,
        "labeled": 0, "boxes": 0,
        "per_class": {i: 0 for i in range(7)},
    }

    for split in ["train", "val"]:
        img_dir = SRC_DATASET / split / "images"
        lbl_dir = SRC_DATASET / split / "labels"
        if not img_dir.exists():
            continue

        results = model.predict(
            source=str(img_dir), conf=min(CONF_THRESHOLD, CONF_HEALTHY),
            iou=0.5, imgsz=640, device=0, stream=True, verbose=False,
        )

        for r in results:
            img_path = Path(r.path)
            img_name = img_path.stem
            stats["total"] += 1

            if img_name in exclude:
                stats["skipped"] += 1
                continue

            lbl_path = lbl_dir / f"{img_name}.txt"
            if lbl_path.exists():
                lines = lbl_path.read_text().strip().splitlines()
                if lines and lines[0].strip():
                    c8 = int(lines[0].split()[0])
                    if c8 not in MAP_8TO7:
                        stats["leaf_blight"] += 1
                        continue

            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes
            pred_cls = boxes.cls.cpu().numpy().astype(int)
            pred_conf = boxes.conf.cpu().numpy()
            pred_xywhn = boxes.xywhn.cpu().numpy()

            kept = []
            for i in range(len(pred_cls)):
                c = pred_cls[i]
                if c < 0 or c >= 7:
                    continue
                thresh = CONF_HEALTHY if c == 6 else CONF_THRESHOLD
                if pred_conf[i] >= thresh:
                    kept.append(i)

            if kept:
                stats["labeled"] += 1
                for i in kept:
                    stats["boxes"] += 1
                    stats["per_class"][pred_cls[i]] += 1

                shutil.copy2(img_path, OUT_DIR / split / "images" / img_path.name)
                with open(OUT_DIR / split / "labels" / f"{img_name}.txt", "w") as f:
                    for i in kept:
                        cx, cy, w, h = pred_xywhn[i]
                        f.write(f"{pred_cls[i]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"\n{'='*60}")
    print("V5 伪标注完成")
    print(f"{'='*60}")
    print(f"处理: {stats['total']} | 跳过: {stats['skipped']} | LeafBlight: {stats['leaf_blight']}")
    print(f"有伪标注: {stats['labeled']} | 总框: {stats['boxes']} | 框/图: {stats['boxes']/max(stats['labeled'],1):.2f}")
    print("各类框数:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}: {stats['per_class'][i]}")

    for split in ["train", "val"]:
        imgs = len(list((OUT_DIR / split / "images").glob("*")))
        print(f"  {split}: {imgs} images")

    (OUT_DIR / "data.yaml").write_text(f"""path: {OUT_DIR}
train: train/images
val: val/images
nc: 7
names:
  0: Brown_Rust
  1: Yellow_Rust
  2: Black_Rust
  3: Septoria
  4: Powdery_Mildew
  5: Fusarium_Head_Blight
  6: Healthy_Wheat
""", encoding="utf-8")
    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
