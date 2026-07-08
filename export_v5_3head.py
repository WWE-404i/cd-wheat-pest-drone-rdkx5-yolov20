"""
导出 V5 模型 ONNX — 3 特征图输出（不含 NMS）
RDK X5 BPU 部署用，后处理（decode + NMS）在 CPU 侧执行

输出：
  output0: [1, 11, 80, 80]   stride 8
  output1: [1, 11, 40, 40]   stride 16
  output2: [1, 11, 20, 20]   stride 32
  其中 11 = 4 (bbox) + 7 (class scores)
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
MODEL_PATH = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best.pt"
OUT_PATH = PROJECT / "runs" / "wheat_7class_v5" / "weights" / "best_3head.onnx"


class YOLO3HeadWrapper(nn.Module):
    """包装 ultralytics DetectionModel，输出 3 个独立特征图"""

    def __init__(self, model):
        super().__init__()
        self.model = model  # DetectionModel

    def forward(self, x):
        """返回 3 个特征图 [B,11,80,80], [B,11,40,40], [B,11,20,20]"""
        # 前 22 层 = backbone + neck
        # 第 23 层 = Detect head
        detect = self.model.model[-1]  # Detect 模块
        neck = self.model.model[:-1]   # backbone + neck

        # 提取 neck 输出的 3 层特征图
        feats = []
        # 先跑 backbone + neck
        # 需要处理 Detect 需要的中间层输出
        # ultralytics 的 forward 会把 neck 输出传给 Detect
        # 这里直接调用完整 forward 会走到 _inference，需要绕过

        # 方案：手动跑 backbone + neck，拿到 3 个特征图
        y = x
        for i, m in enumerate(self.model.model):
            if i == len(self.model.model) - 1:  # Detect 层
                # m 是 Detect，手动调用 conv 但不做 concat/resize
                outputs = []
                for j in range(m.nl):
                    # x 已经是 3 个尺度的 neck 输出
                    x_j = y[j] if isinstance(y, list) else y
                    if isinstance(y, list) and j < len(y):
                        x_j = y[j]
                    box = m.cv2[j](x_j)     # [B, 4, H, W]  (reg_max=1)
                    cls = m.cv3[j](x_j)     # [B, 7, H, W]
                    out = torch.cat([box, cls], dim=1)  # [B, 11, H, W]
                    outputs.append(out)
                return outputs[0], outputs[1], outputs[2]
            elif isinstance(m, nn.Upsample):
                y = m(y)
            elif hasattr(m, 'f') and m.f != -1:
                # Concat / other layers with from-layer connections
                y = [y] if not isinstance(y, list) else y
                y.append(y[m.f] if isinstance(y[m.f], int) and y[m.f] < len(y) else None)
                # ultralytics handles this internally
                y = m(y)
            else:
                y = m(y)

        return y  # shouldn't reach here


def export():
    print("Loading model...")
    yolo = YOLO(str(MODEL_PATH))
    model = yolo.model
    model.eval()

    detect = model.model[-1]
    print(f"Detect: strides={detect.stride.tolist()}, nc={detect.nc}, reg_max={detect.reg_max}")

    # 方法：torch.onnx.export 时使用自定义 forward
    # 直接用原始模型的 forward_head，但需要 per-scale 输出

    # 最简洁方式：直接用 torch.onnx.export 导出，
    # forward 函数从 neck 拿到 3 个特征图后分别过 cv2/cv3

    dummy = torch.randn(1, 3, 640, 640)

    # 验证原始模型输出
    with torch.no_grad():
        # 用训练模式拿到 raw preds
        model.train()  # training mode → forward_head → raw dict
        preds = model(dummy)
        model.eval()
        if isinstance(preds, dict):
            boxes = preds['boxes']  # [1, 4, 8400]
            scores = preds['scores']  # [1, 7, 8400]
            feats = preds['feats']   # 3 tensors
            for i, f in enumerate(feats):
                print(f"  feat[{i}]: {f.shape}")
            print(f"  boxes (concat): {boxes.shape}")
            print(f"  scores (concat): {scores.shape}")

    # 获取 per-scale 输出
    print("\nPer-scale outputs (cv2+3 applied separately):")
    with torch.no_grad():
        x = dummy
        neck = model.model[:-1]
        detect = model.model[-1]
        feats = model.model.forward(x)  # uses the built-in forward

    # 更直接的办法：forward the model in training mode to get feats
    model.train()
    with torch.no_grad():
        preds = model(dummy)
    model.eval()
    feats = preds['feats']

    print(f"\nFeats from training mode:")
    for i, f in enumerate(feats):
        print(f"  feat[{i}]: {f.shape}")

    # Per-scale Apply cv2/cv3
    print("\nPer-scale detection outputs:")
    per_scale = []
    for i in range(detect.nl):
        box = detect.cv2[i](feats[i])  # [B, 4*reg_max, H, W]
        cls = detect.cv3[i](feats[i])  # [B, nc, H, W]
        out_i = torch.cat([box, cls], dim=1)  # [B, 11, H, W]
        per_scale.append(out_i)
        print(f"  output[{i}]: {out_i.shape}")

    # === Export ONNX ===
    print(f"\nExporting ONNX to {OUT_PATH}...")

    # 用模型原始 forward 导出（train mode → raw feats → per-scale outputs）
    # 包装成一个干净的前向函数

    class ExportWrapper(nn.Module):
        def __init__(self, full_model):
            super().__init__()
            # full_model 是 ultralytics 的 DetectionModel
            self.backbone_neck = full_model.model[:-1]  # layers 0-22
            self.detect = full_model.model[-1]           # Detect layer

        def forward(self, x):
            # 跑 backbone+neck
            y = self.backbone_neck(x)
            # y 应该是 3 个 list（ultralytics 的 neck 输出是 list 或 tuple）
            outputs = []
            for j in range(self.detect.nl):
                box = self.detect.cv2[j](y[j])
                cls = self.detect.cv3[j](y[j])
                out = torch.cat([box, cls], dim=1)  # [B, 11, H, W]
                outputs.append(out)
            return outputs[0], outputs[1], outputs[2]

    # 检查 neck 是否真的输出 list
    model.eval()
    with torch.no_grad():
        neck_out = model.model[:-1](dummy)
    print(f"  Neck output type: {type(neck_out)}")
    if isinstance(neck_out, (list, tuple)):
        for i, f in enumerate(neck_out):
            print(f"    neck[{i}]: {f.shape}")

    wrapper = ExportWrapper(model)
    wrapper.eval()

    # 验证 wrapper 输出
    with torch.no_grad():
        o0, o1, o2 = wrapper(dummy)
        print(f"  Wrapper output[0]: {o0.shape}")
        print(f"  Wrapper output[1]: {o1.shape}")
        print(f"  Wrapper output[2]: {o2.shape}")

    # Export
    torch.onnx.export(
        wrapper,
        dummy,
        str(OUT_PATH),
        opset_version=11,
        input_names=['images'],
        output_names=['output0', 'output1', 'output2'],
        dynamic_axes=None,
        do_constant_folding=True,
    )

    print(f"\nONNX exported: {OUT_PATH}")

    # === Verify with onnxruntime ===
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(OUT_PATH))
        ort_inputs = {'images': dummy.numpy()}
        ort_outs = sess.run(None, ort_inputs)
        print(f"\nONNX Runtime verification:")
        for i, out in enumerate(ort_outs):
            print(f"  output[{i}]: {out.shape}  min={out.min():.4f}  max={out.max():.4f}")

        # Check with original per_scale outputs
        for i in range(3):
            diff = np.abs(ort_outs[i] - per_scale[i].numpy()).max()
            print(f"  output[{i}] max diff vs PyTorch: {diff:.6f}")
    except Exception as e:
        print(f"  onnxruntime check skipped: {e}")

    print("\nDone!")
    file_size = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")


if __name__ == "__main__":
    import numpy as np
    export()
