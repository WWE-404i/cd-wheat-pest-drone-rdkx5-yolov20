# cd-wheat-pest-drone-rdkx5-yolov20

A complete open-source project for quadcopter wheat pest and disease inspection based on Horizon RDK X5, YOLOv20, and Raspberry Pi camera. Includes model training, RKNN quantization, onboard real-time inference, and flight controller serial communication code.

---

## 📁 Project Structure

```
├── 主流程/         🚀 One-click launcher
├── 标注工具/       🏷️  Image annotation (SAM / Grounding DINO / Florence)
├── 伪标签生成/     🤖 Auto-labeling unlabeled data with trained models
├── 数据处理/       📦 Dataset merging, format conversion, golden set selection
├── 训练脚本/       🎯 YOLO training (v3~v9 multiple iterations)
├── 验证检查/       ✅ Model prediction, inspection, validation
├── RDK部署/        📱 Horizon RDK X5 export & quantization
├── 可视化/         📊 Annotation visualization, heatmaps, webcam real-time detection
├── 测试评估/       🧪 Grounding DINO benchmarking & calibration
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### One-Click Run

```bash
# Lite: SAM annotation → dataset split → YOLO training
python 主流程/pipeline.py

# Full: Calibration → annotation → filtering → training (run overnight)
python 主流程/pipeline_full.py
```

Both scripts support **checkpoint resume** — just re-run after interruption to continue.

---

## 📂 Directory Details

### 🚀 Main Pipeline

| File | Description |
|------|-------------|
| `pipeline.py` | One-stop pipeline: SAM auto-annotation → train/val split → YOLOv26n training, with checkpoint resume |
| `pipeline_full.py` | Full pipeline: GD threshold calibration → full annotation (13K+ images, 7 classes) → image/box filtering → YOLO training |

### 🏷️ Annotation Tools

Leverage large models to assist wheat disease image annotation, reducing manual labeling effort:

| File | Description |
|------|-------------|
| `annotate_grounding_dino.py` | Grounding DINO auto-annotation |
| `annotate_gd_tile.py` / `_fast.py` | Tiled annotation for large images |
| `annotate_gd_v2.py` / `v3.py` | Improved annotation pipelines |
| `annotate_gd_full.py` | Full dataset annotation |
| `annotate_florence.py` | Microsoft Florence-2 model annotation |
| `annotate_sam_refine.py` | SAM bounding box refinement |
| `annotate_with_vlm.py` | VLM (Vision Language Model) annotation |
| `label_tool.py` / `v2.py` | Manual annotation tools |
| `review_tool.py` | Annotation review tool |
| `review_annotations.py` | Batch annotation quality audit |
| `analyze_box_sizes.py` | Bounding box size distribution analysis |

### 🤖 Pseudo-Label Generation

Auto-label new data with trained models to expand the training set:

| File | Description |
|------|-------------|
| `pseudo_label_v1~v3.py` | Pseudo-label generation iterations |
| `pseudo_label_v5.py` | Latest pseudo-label generation |

### 📦 Data Processing

| File | Description |
|------|-------------|
| `convert_archive_to_yolo.py` | Convert raw data to YOLO format |
| `convert_golden_7class.py` | Golden set 7-class conversion |
| `merge_v6~v9_dataset.py` | Dataset merging across versions |
| `merge_golden.py` | Golden annotation set merging |
| `select_golden.py` | Golden standard sample selection |
| `data_7class.yaml` | 7-class dataset configuration |

### 🎯 Training Scripts

| File | Description |
|------|-------------|
| `train_baseline.py` | Baseline model training |
| `train_v3.py ~ train_v9.py` | Iterative training versions (v6b/c/d are v6 variants) |
| `train_manual_v1.py` | Training with purely manual annotations |
| `merge_and_train_v2.py` | Dataset merging + training in one step |

### ✅ Validation & Inspection

| File | Description |
|------|-------------|
| `_check_golden.py` | Golden set annotation quality check |
| `_check_v2.py` / `_check_history.py` | Dataset correctness checks |
| `_check_rust_result.py` | Rust disease detection analysis |
| `_check_reviewed.py` | Re-audit reviewed annotations |
| `_find_rust_images.py` | Filter images containing rust diseases |
| `_v5_predict.py` / `_v5_predict_hd.py` | V5 model prediction (SD/HD) |
| `_v9_check.py` | V9 model inspection |
| `_verify_images.py` | Image integrity verification |
| `check_v6b.py ~ check_v9.py` | Per-version model accuracy checks |
| `validate_visual.py` | Visual validation of results |

### 📱 RDK Deployment

| File | Description |
|------|-------------|
| `export_v5_3head.py` | Export RDK X5 three-head detection model |
| `rdk_x5_ptq_calib.yaml` | RDK X5 PTQ quantization calibration config |

### 📊 Visualization

| File | Description |
|------|-------------|
| `visualize_labels.py` | Bounding box visualization |
| `visualize_pseudo.py` | Pseudo-label effect visualization |
| `create_reference_panel.py` | Reference image panel creation |
| `download_reference_images.py` | Reference image downloader |
| `extract_reference_crops.py` | Crop target regions from reference images |
| `make_reference_chart.py` | Reference comparison chart |
| `find_hard_cases.py` | Hard case mining |
| `find_hard_cases_supplement.py` | Supplementary hard case mining |
| `pick_to_label.py` | Select images for manual labeling |
| `webcam_v5.py` | Real-time webcam detection demo |

### 🧪 Testing & Evaluation

| File | Description |
|------|-------------|
| `benchmark_gd_tile.py` | Grounding DINO tiled inference benchmark |
| `calibrate_gd.py` | Grounding DINO threshold calibration |
| `test_gd_tile.py` / `_resize.py` / `_v3.py` | Tiled/resize/new version annotation tests |
| `test_fast_vs_orig.py` | Fast vs. original comparison test |

---

## 🔬 Detection Targets (7 Classes)

1. Brown Rust
2. Yellow Rust
3. Black Rust
4. Septoria
5. Mildew (Powdery Mildew)
6. Fusarium Head Blight
7. Leaf Blight

---

## 🛠️ Tech Stack

- **Object Detection**: YOLOv20 / YOLOv26n
- **Auto Annotation**: SAM, Grounding DINO, Florence-2
- **Edge Deployment**: Horizon RDK X5, RKNN quantization
- **Camera**: Raspberry Pi camera real-time inference
