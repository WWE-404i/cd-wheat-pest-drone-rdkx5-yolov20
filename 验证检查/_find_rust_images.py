"""从精标+审核中找出含锈病(0/1/2)的图片"""
from pathlib import Path
import shutil

GOLDEN = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_disease_golden")
REVIEWED = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_pseudo_v5_reviewed")
OUT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\rust_review")
SRC_IMAGES = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_disease_8class")

CLS = ["Brown_Rust","Yellow_Rust","Black_Rust","Septoria","Powdery_Mildew","Fusarium_HB","Healthy"]
RUST_CLASSES = {0, 1, 2}

def find_image(stem):
    for s in ["train", "val"]:
        p = SRC_IMAGES / s / "images" / f"{stem}.jpg"
        if p.exists():
            return p
    return None

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

seen = set()
total = 0
stats = {"golden": 0, "reviewed": 0}

for source_name, src_dir in [("golden", GOLDEN), ("reviewed", REVIEWED)]:
    # golden有train/val子目录，reviewed是扁平labels/
    if source_name == "golden":
        label_dirs = []
        for split in ["train", "val"]:
            d = src_dir / split / "labels"
            if d.exists():
                label_dirs.append(d)
    else:
        label_dirs = [src_dir / "labels"] if (src_dir / "labels").exists() else []

    for lbl_dir in label_dirs:
        for lbl in sorted(lbl_dir.glob("*.txt")):
            stem = lbl.stem
            if stem in seen:
                continue

            classes = set()
            for line in lbl.read_text().strip().splitlines():
                if not line.strip():
                    continue
                classes.add(int(line.split()[0]))

            # 必须有锈病
            if not (classes & RUST_CLASSES):
                continue

            seen.add(stem)
            stats[source_name] += 1
            total += 1

            # 复制标签
            shutil.copy2(lbl, OUT / "labels" / f"{stem}.txt")

print(f"含锈病图片:")
print(f"  精标: {stats['golden']} 张")
print(f"  审核: {stats['reviewed']} 张")
print(f"  合计: {total} 张")
print(f"\n标签已复制到: {OUT}/labels/")

# 按锈病种类细分
both_0_2 = []
has_0 = []
has_1 = []
has_2 = []
multi_rust = []

for lbl in sorted((OUT / "labels").glob("*.txt")):
    classes = set()
    counts = {}
    for line in lbl.read_text().strip().splitlines():
        if not line.strip():
            continue
        c = int(line.split()[0])
        classes.add(c)
        counts[c] = counts.get(c, 0) + 1

    rust_in_img = classes & RUST_CLASSES
    if len(rust_in_img) >= 2:
        multi_rust.append((lbl.stem, rust_in_img, counts))
    if 0 in classes and 2 in classes:
        both_0_2.append((lbl.stem, counts))

print(f"\n多种锈病共存: {len(multi_rust)} 张")
for stem, rusts, cts in sorted(multi_rust):
    rust_names = [CLS[c] for c in rusts]
    print(f"  {stem}: 锈病={rust_names}  全部={dict(cts)}")

print(f"\n同时有0和2: {len(both_0_2)} 张")
for stem, cts in sorted(both_0_2):
    print(f"  {stem}: {cts}")
