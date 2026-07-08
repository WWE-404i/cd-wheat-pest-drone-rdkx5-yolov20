"""
黄金标注工具 (tkinter版) — 鼠标画框 + 键盘选类 → YOLO格式
用法: python label_tool.py
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
from pathlib import Path
import json
import sys

# ========== 配置 ==========
SOURCE_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\golden_set\images")
LABEL_DIR = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\golden_set\labels")
CHECKPOINT_FILE = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型\golden_set\checkpoint.json")

CLASS_NAMES = [
    "0:Brown Rust", "1:Yellow Rust", "2:Black Rust", "3:Septoria",
    "4:Mildew", "5:Fusarium HB", "6:Healthy(no box)",
]
COLORS = [
    "#FF0000", "#FF8800", "#CC6600", "#00CC00",
    "#00CCCC", "#0066FF", "#888888",
]

DISPLAY_SIZE = 1100


class LabelTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Golden Label Tool")
        self.root.geometry(f"{DISPLAY_SIZE+240}x{DISPLAY_SIZE+60}")

        # 图片列表
        self.img_files = []
        files = list(SOURCE_DIR.glob("*.jpg")) + list(SOURCE_DIR.glob("*.png")) + list(SOURCE_DIR.glob("*.jpeg"))
        if not files:
            messagebox.showerror("Error", f"{SOURCE_DIR} 中没有图片!\n请先运行 select_golden.py")
            sys.exit(1)
        self.img_files = sorted(files, key=lambda x: (x.name.split('_')[0], x.name))

        # 状态
        self.ckpt = self._load_checkpoint()
        self.idx = self.ckpt.get("current_idx", 0)
        if self.idx >= len(self.img_files):
            self.idx = 0

        self.current_cls = 0
        self.boxes = []          # 显示坐标 [(cls, x1, y1, x2, y2), ...]
        self.drawing = False
        self.start_x = self.start_y = 0
        self.tmp_rect = None
        self.pil_img = None
        self.tk_img = None
        self.scale = 1.0
        self.offset_x = self.offset_y = 0

        # ==== UI ====
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧画布
        self.canvas = tk.Canvas(main_frame, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                bg='#1a1a1a', cursor='crosshair')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # 右侧面板
        panel = ttk.Frame(main_frame, width=220)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # 进度
        ttk.Label(panel, text="进度", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
        self.progress_label = ttk.Label(panel, text="")
        self.progress_label.pack(anchor=tk.W)

        self.filename_label = ttk.Label(panel, text="", wraplength=210, font=('', 8))
        self.filename_label.pack(anchor=tk.W, pady=(0,10))

        # 类别选择
        ttk.Label(panel, text="当前类别 (0-7切换)", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
        self.cls_var = tk.IntVar(value=0)
        for i, name in enumerate(CLASS_NAMES):
            rb = ttk.Radiobutton(panel, text=name, variable=self.cls_var, value=i,
                                 command=lambda i=i: self._switch_cls(i))
            rb.pack(anchor=tk.W, padx=5)
        panel.bind_all(str(self.current_cls), lambda e: self._switch_cls(int(e.char)))
        for i in range(7):
            panel.bind_all(str(i), lambda e, i=i: self._switch_cls(i))

        # 框信息
        ttk.Label(panel, text="", font=('', 1)).pack()
        self.box_count_label = ttk.Label(panel, text="框数: 0", font=('', 10, 'bold'))
        self.box_count_label.pack(anchor=tk.W)

        # 按钮
        ttk.Label(panel, text="", font=('', 1)).pack()
        ttk.Button(panel, text="下一张 (N/Space)", command=self._next).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="上一张 (←)", command=self._prev).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="删最后一个框 (D)", command=self._delete_box).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="清空所有框 (C)", command=self._clear_boxes).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="手动保存 (S)", command=self._save).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="退出 (Q)", command=self._quit).pack(fill=tk.X, pady=(20,2))

        # 缩放提示
        self.info_label = ttk.Label(panel, text="", font=('', 8), foreground='gray')
        self.info_label.pack(anchor=tk.W, pady=(10,0))

        # 键盘绑定
        self.root.bind('<Key>', self._on_key)

        # 加载第一张
        self._load_image()

        self.root.mainloop()

    def _load_checkpoint(self):
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        return {"current_idx": 0, "labeled": []}

    def _save_checkpoint(self):
        ckpt = {"current_idx": self.idx, "labeled": list(set(
            self.ckpt.get("labeled", []) + [self.img_files[self.idx].name]))}
        LABEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(ckpt, f, indent=2)

    def _load_image(self):
        """加载当前图片 (切换图片时调用，不会从磁盘恢复标签)"""
        self.canvas.delete("all")
        self.boxes = []

        img_path = self.img_files[self.idx]
        try:
            self.pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"无法读取 {img_path.name}: {e}")
            return

        # 缩放适配画布
        w, h = self.pil_img.size
        self.scale = min(DISPLAY_SIZE / max(w, h), 1.0)
        self.new_w, self.new_h = int(w * self.scale), int(h * self.scale)
        self.offset_x = (DISPLAY_SIZE - self.new_w) // 2
        self.offset_y = (DISPLAY_SIZE - self.new_h) // 2

        resized = self.pil_img.resize((self.new_w, self.new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_img)

        # 从磁盘恢复已有标注 (仅在首次打开时)
        lbl_path = LABEL_DIR / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy = float(parts[1]), float(parts[2])
                    bw, bh = float(parts[3]), float(parts[4])
                    x1 = int((cx - bw/2) * self.new_w) + self.offset_x
                    y1 = int((cy - bh/2) * self.new_h) + self.offset_y
                    x2 = int((cx + bw/2) * self.new_w) + self.offset_x
                    y2 = int((cy + bh/2) * self.new_h) + self.offset_y
                    self.boxes.append((cls_id, x1, y1, x2, y2))

        self._redraw()

    def _redraw(self):
        """重绘所有框 (不重新加载图片)"""
        # 清除旧的框图形
        self.canvas.delete("box")
        self.canvas.delete("tmp")
        for cls_id, x1, y1, x2, y2 in self.boxes:
            self._draw_box(cls_id, x1, y1, x2, y2)
        self._update_info()

    def _draw_box(self, cls_id, x1, y1, x2, y2):
        color = COLORS[cls_id % len(COLORS)]
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
        label = f"{cls_id}:{CLASS_NAMES[cls_id].split(':')[1][:8]}"
        self.canvas.create_rectangle(x1, y1-16, x1+len(label)*7+4, y1, fill=color, outline=color, tags="box")
        self.canvas.create_text(x1+2, y1-9, text=label, anchor=tk.W, fill="white", font=('', 8), tags="box")

    def _on_mouse_down(self, event):
        self.drawing = True
        self.start_x, self.start_y = event.x, event.y

    def _on_mouse_drag(self, event):
        if self.drawing:
            self.canvas.delete("tmp")
            self.tmp_rect = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline=COLORS[self.current_cls], width=1, dash=(4,4), tags="tmp")

    def _on_mouse_up(self, event):
        self.drawing = False
        self.canvas.delete("tmp")
        self.tmp_rect = None

        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        bw, bh = x2 - x1, y2 - y1

        if bw > 5 and bh > 5:
            self.boxes.append((self.current_cls, x1, y1, x2, y2))
            self._draw_box(self.current_cls, x1, y1, x2, y2)
            # 像素尺寸提示
            px_w = int(bw / self.scale)
            px_h = int(bh / self.scale)
            self.info_label.config(text=f"框: {px_w}x{px_h}px | {CLASS_NAMES[self.current_cls]}")
        self._update_info()

    def _switch_cls(self, cls_id):
        self.current_cls = cls_id
        self.cls_var.set(cls_id)
        self.root.title(f"Golden Label Tool — [{self.current_cls}] {CLASS_NAMES[self.current_cls]}")

    def _on_key(self, event):
        key = event.keysym.lower()
        if key in ('n', 'space', 'right'):
            self._next()
        elif key == 'left':
            self._prev()
        elif key == 'd':
            self._delete_box()
        elif key == 'c':
            self._clear_boxes()
        elif key == 's':
            self._save()
        elif key in ('q', 'escape'):
            self._quit()
        elif key in '0123456':
            self._switch_cls(int(key))

    def _update_info(self):
        img_path = self.img_files[self.idx]
        short = min(self.pil_img.size)
        self.progress_label.config(text=f"{self.idx+1}/{len(self.img_files)}")
        self.filename_label.config(text=f"{img_path.name}\n{self.pil_img.size[0]}x{self.pil_img.size[1]}px (短边{short})")
        self.box_count_label.config(text=f"框数: {len(self.boxes)}")
        self.root.title(f"Golden Label Tool — [{self.current_cls}] {CLASS_NAMES[self.current_cls]}")

        # 统计已标注
        labeled = len(list(LABEL_DIR.glob("*.txt")))
        self.info_label.config(text=f"已标注: {labeled}/{len(self.img_files)}")

    def _save(self):
        img_path = self.img_files[self.idx]
        lbl_path = LABEL_DIR / f"{img_path.stem}.txt"
        lbl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lbl_path, 'w') as f:
            for cls_id, x1, y1, x2, y2 in self.boxes:
                cx = ((x1 + x2) / 2 - self.offset_x) / self.new_w
                cy = ((y1 + y2) / 2 - self.offset_y) / self.new_h
                bw = (x2 - x1) / self.new_w
                bh = (y2 - y1) / self.new_h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.001, min(1.0, bw))
                bh = max(0.001, min(1.0, bh))
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        self._save_checkpoint()
        self.info_label.config(text=f"已保存 {img_path.name}")

    def _next(self):
        self._save()
        self.idx = min(self.idx + 1, len(self.img_files) - 1)
        self._load_image()

    def _prev(self):
        self._save()
        self.idx = max(self.idx - 1, 0)
        self._load_image()

    def _delete_box(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw()

    def _clear_boxes(self):
        self.boxes.clear()
        self._redraw()

    def _quit(self):
        self._save()
        labeled = len(list(LABEL_DIR.glob("*.txt")))
        messagebox.showinfo("退出", f"进度已保存。\n已标注: {labeled}/{len(self.img_files)}")
        self.root.destroy()


def main():
    LabelTool()


if __name__ == "__main__":
    main()
