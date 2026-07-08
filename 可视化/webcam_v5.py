"""V5 实时摄像头检测"""
import cv2
# ultralytics 会劫持 cv2.imshow，先保留原始引用
_orig_imshow = cv2.imshow
from ultralytics import YOLO
cv2.imshow = _orig_imshow  # 恢复原始 imshow
from pathlib import Path

MODEL_PATH = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\runs\wheat_7class_v5\weights\best.pt")

CLS_NAMES = {
    0: ("Brown Rust", "褐锈病"),
    1: ("Yellow Rust", "黄锈病"),
    2: ("Black Rust", "黑锈病"),
    3: ("Septoria", "壳针孢病"),
    4: ("Powdery Mildew", "白粉病"),
    5: ("Fusarium Head Blight", "赤霉病"),
    6: ("Healthy Wheat", "健康小麦"),
}

COLORS = [
    (0, 0, 255),     # 0 红
    (0, 255, 255),   # 1 黄
    (255, 0, 0),     # 2 蓝
    (0, 255, 0),     # 3 绿
    (255, 0, 255),   # 4 紫
    (255, 255, 0),   # 5 青
    (128, 128, 128), # 6 灰
]

print("Loading V5 model...")
model = YOLO(str(MODEL_PATH))

print("Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Running. Press 'q' to quit, 's' to screenshot.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.3, iou=0.5, imgsz=640)

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                en, cn = CLS_NAMES.get(cls_id, (f"cls{cls_id}", f"未知{cls_id}"))
                color = COLORS[cls_id % len(COLORS)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cn} ({en}) {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS
    speed = results[0].speed if results else {}
    fps_text = f"FPS: {1000/speed['inference']:.0f}" if speed.get('inference') else ""
    if fps_text:
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Wheat Disease Detection - V5", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("webcam_screenshot.jpg", frame)
        print("Saved: webcam_screenshot.jpg")

cap.release()
cv2.destroyAllWindows()
print("Done.")
