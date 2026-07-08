from pathlib import Path
SRC = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\wheat_disease_8class")
RUST = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\rust_review\labels")
missing = []
for lbl in RUST.glob("*.txt"):
    found = False
    for s in ["train","val"]:
        if (SRC / s / "images" / f"{lbl.stem}.jpg").exists():
            found = True; break
    if not found:
        missing.append(lbl.stem)
print(f"总数: {len(list(RUST.glob('*.txt')))}")
print(f"缺图: {len(missing)}")
if missing:
    for m in missing[:10]:
        print(f"  {m}")
