# cd-wheat-pest-drone-rdkx5-yolov20

基于地平线 RDK X5、YOLOv20、树莓派摄像头的四旋翼小麦病虫害巡检无人机完整开源工程，包含模型训练、RKNN 量化、机载实时推理、飞控串口交互代码。

---

## 📁 项目结构

```
├── 主流程/         🚀 一键启动入口
├── 标注工具/       🏷️  图像标注（SAM / Grounding DINO / Florence）
├── 伪标签生成/     🤖 模型自动标注未标记数据
├── 数据处理/       📦 数据集合并、格式转换、黄金集筛选
├── 训练脚本/       🎯 YOLO 训练（v3~v9 多个迭代版本）
├── 验证检查/       ✅ 模型预测、检查、验证
├── RDK部署/        📱 地平线 RDK X5 导出与量化
├── 可视化/         📊 标注可视化、热力图、Webcam 实时检测
├── 测试评估/       🧪 Grounding DINO 基准测试与校准
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 快速开始

### 一键运行

```bash
# 精简版：SAM 标注 → 切分数据集 → YOLO 训练
python 主流程/pipeline.py

# 完整版：校准 → 标注 → 过滤 → 训练（睡前跑，醒来收结果）
python 主流程/pipeline_full.py
```

两个脚本都支持**断点续传**，中途中断重新运行即可继续。

---

## 📂 各目录说明

### 🚀 主流程

| 文件 | 说明 |
|------|------|
| `pipeline.py` | 一条龙流水线：SAM 自动标注 → 切分训练集 → YOLOv26n 训练，支持断点续传 |
| `pipeline_full.py` | 完整流水线：GD 阈值校准 → 全量标注(13K+图片, 7类) → 图片/框过滤 → YOLO 训练 |

### 🏷️ 标注工具

使用大模型辅助标注小麦病虫害图像，减少人工标注工作量：

| 文件 | 说明 |
|------|------|
| `annotate_grounding_dino.py` | Grounding DINO 自动标注 |
| `annotate_gd_tile.py` / `_fast.py` | 分块标注大图 |
| `annotate_gd_v2.py` / `v3.py` | 标注流程改进版 |
| `annotate_gd_full.py` | 全量数据标注 |
| `annotate_florence.py` | Microsoft Florence-2 模型标注 |
| `annotate_sam_refine.py` | SAM 精修标注框 |
| `annotate_with_vlm.py` | VLM（视觉语言模型）标注 |
| `label_tool.py` / `v2.py` | 手工标注工具 |
| `review_tool.py` | 标注结果复核工具 |
| `review_annotations.py` | 批量审核标注质量 |
| `analyze_box_sizes.py` | 分析标注框尺寸分布 |

### 🤖 伪标签生成

用已训练模型对新数据自动打标签，扩充训练集：

| 文件 | 说明 |
|------|------|
| `pseudo_label_v1~v3.py` | 伪标签生成迭代版 |
| `pseudo_label_v5.py` | 伪标签生成最新版 |

### 📦 数据处理

| 文件 | 说明 |
|------|------|
| `convert_archive_to_yolo.py` | 原始数据转 YOLO 格式 |
| `convert_golden_7class.py` | 黄金集 7 分类转换 |
| `merge_v6~v9_dataset.py` | 各版本数据集合并 |
| `merge_golden.py` | 黄金标注集合并 |
| `select_golden.py` | 筛选黄金标准样本 |
| `data_7class.yaml` | 7 分类数据集配置 |

### 🎯 训练脚本

| 文件 | 说明 |
|------|------|
| `train_baseline.py` | 基线模型训练 |
| `train_v3.py ~ train_v9.py` | 各迭代版本训练（v6b/c/d 为 v6 的变体） |
| `train_manual_v1.py` | 纯手工标注数据训练 |
| `merge_and_train_v2.py` | 合并数据集 + 训练一步完成 |

### ✅ 验证检查

| 文件 | 说明 |
|------|------|
| `_check_golden.py` | 检查黄金集标注质量 |
| `_check_v2.py` / `_check_history.py` | 检查数据集正确性 |
| `_check_rust_result.py` | 锈病检测结果分析 |
| `_check_reviewed.py` | 复核已审核标注 |
| `_find_rust_images.py` | 筛选含锈病图片 |
| `_v5_predict.py` / `_v5_predict_hd.py` | v5 模型预测（标清/高清） |
| `_v9_check.py` | v9 模型检查 |
| `_verify_images.py` | 图片完整性校验 |
| `check_v6b.py ~ check_v9.py` | 各版本模型精度检查 |
| `validate_visual.py` | 可视化验证结果 |

### 📱 RDK 部署

| 文件 | 说明 |
|------|------|
| `export_v5_3head.py` | 导出 RDK X5 三检测头模型 |
| `rdk_x5_ptq_calib.yaml` | RDK X5 PTQ 量化校准配置 |

### 📊 可视化

| 文件 | 说明 |
|------|------|
| `visualize_labels.py` | 标注框可视化 |
| `visualize_pseudo.py` | 伪标签效果可视化 |
| `create_reference_panel.py` | 创建参考图面板 |
| `download_reference_images.py` | 下载参考图片 |
| `extract_reference_crops.py` | 裁剪参考图目标区域 |
| `make_reference_chart.py` | 制作参考对比图 |
| `find_hard_cases.py` | 查找困难样本 |
| `find_hard_cases_supplement.py` | 补充查找困难样本 |
| `pick_to_label.py` | 挑选待标注图片 |
| `webcam_v5.py` | 摄像头实时检测 Demo |

### 🧪 测试评估

| 文件 | 说明 |
|------|------|
| `benchmark_gd_tile.py` | Grounding DINO 分块推理性能测试 |
| `calibrate_gd.py` | Grounding DINO 阈值校准 |
| `test_gd_tile.py` / `_resize.py` / `_v3.py` | 分块/缩放/新版标注测试 |
| `test_fast_vs_orig.py` | 快速版 vs 原版对比测试 |

---

## 🔬 检测目标（7 类）

1. Brown Rust（叶锈病）
2. Yellow Rust（条锈病）
3. Black Rust（秆锈病）
4. Septoria（叶斑病）
5. Mildew（白粉病）
6. Fusarium Head Blight（赤霉病）
7. Leaf Blight（叶枯病）

---

## 🛠️ 技术栈

- **目标检测**: YOLOv20 / YOLOv26n
- **自动标注**: SAM、Grounding DINO、Florence-2
- **边缘部署**: 地平线 RDK X5、RKNN 量化
- **摄像头**: 树莓派摄像头实时推理
