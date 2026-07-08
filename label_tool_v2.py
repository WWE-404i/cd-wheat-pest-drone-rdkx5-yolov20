"""
手工标注工具 v2 — 7 类系统，支持继续标注
用法: python label_tool_v2.py

7 类:
  0: Brown Rust    1: Yellow Rust   2: Black Rust    3: Septoria
  4: Powdery Mildew  5: Fusarium HB   6: Healthy(空标签)

操作:
  鼠标拖拽画框 | 数字键 0-6 切换类别
  N/Space → 下一张 | ← → 上一张
  D → 删最后一个框 | C → 清空所有框
  S → 手动保存 | Q → 退出
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import json

# ========== 配置 ==========
PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")

# 图片源：精选待标注集
SOURCE_DIRS = [
    PROJECT / "to_label_v2",  # V3难例挖掘，210张，按类分文件夹
]

# 标注输出目录
LABEL_OUT = PROJECT / "to_label_v2" / "labels"
IMAGES_OUT = PROJECT / "to_label_v2" / "images"

# 断点文件
CHECKPOINT_FILE = PROJECT / "manual_label_checkpoint_v2.json"

# 7 类
CLASS_NAMES = [
    "0:Brown Rust", "1:Yellow Rust", "2:Black Rust", "3:Septoria",
    "4:Powdery Mildew", "5:Fusarium HB", "6:Healthy(no box)",
]
COLORS = [
    "#FF4444", "#FFAA00", "#CC6600", "#00CC00",
    "#00AAAA", "#0066FF", "#888888",
]

DISPLAY_SIZE = 1100
MIN_SHORT_SIDE = 480  # 只列短边 >= 480 的图


class LabelTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Label Tool v2 — 7类")
        self.root.geometry(f"{DISPLAY_SIZE+240}x{DISPLAY_SIZE+60}")

        # 收集图片
        self.img_files = self._collect_images()
        if not self.img_files:
            messagebox.showerror("Error", "没有找到图片! 检查 image_audit/ 目录")
            return

        print(f"共 {len(self.img_files)} 张待标注图片")

        # 状态
        LABEL_OUT.mkdir(parents=True, exist_ok=True)
        IMAGES_OUT.mkdir(parents=True, exist_ok=True)

        self.ckpt = self._load_checkpoint()
        self.idx = self.ckpt.get("current_idx", 0)
        if self.idx >= len(self.img_files):
            self.idx = 0

        self.current_cls = 0
        self.boxes = []
        self.drawing = False
        self.start_x = self.start_y = 0
        self.pil_img = None
        self.tk_img = None
        self.scale = 1.0
        self.offset_x = self.offset_y = 0

        # ==== UI ====
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(main_frame, width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                bg='#1a1a1a', cursor='crosshair')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        panel = ttk.Frame(main_frame, width=220)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        ttk.Label(panel, text="进度", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.progress_label = ttk.Label(panel, text="")
        self.progress_label.pack(anchor=tk.W)

        self.filename_label = ttk.Label(panel, text="", wraplength=210, font=('', 8))
        self.filename_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(panel, text="类别 (键盘 0-6)", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        self.cls_var = tk.IntVar(value=0)
        for i, name in enumerate(CLASS_NAMES):
            rb = ttk.Radiobutton(panel, text=name, variable=self.cls_var, value=i,
                                 command=lambda i=i: self._switch_cls(i))
            rb.pack(anchor=tk.W, padx=5)

        ttk.Label(panel, text="", font=('', 1)).pack()
        self.box_count_label = ttk.Label(panel, text="框数: 0", font=('', 10, 'bold'))
        self.box_count_label.pack(anchor=tk.W)

        ttk.Label(panel, text="", font=('', 1)).pack()
        ttk.Button(panel, text="下一张 (N/Space)", command=self._next).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="上一张 (←)", command=self._prev).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="删最后一个框 (D)", command=self._delete_box).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="清空所有框 (C)", command=self._clear_boxes).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="手动保存 (S)", command=self._save).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="退出 (Q)", command=self._quit).pack(fill=tk.X, pady=(20, 2))

        self.info_label = ttk.Label(panel, text="", font=('', 8), foreground='gray')
        self.info_label.pack(anchor=tk.W, pady=(10, 0))

        self.root.bind('<Key>', self._on_key)
        self._load_image()
        self.root.mainloop()

    def _collect_images(self):
        """从 image_audit 目录收集待标注图片"""
        files = []
        for src_dir in SOURCE_DIRS:
            if not src_dir.exists():
                continue
            for cls_dir in sorted(src_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for f in sorted(cls_dir.iterdir()):
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        try:
                            with Image.open(f) as img:
                                w, h = img.size
                            if min(w, h) >= MIN_SHORT_SIDE:
                                files.append(f)
                        except Exception:
                            continue
        return sorted(files, key=lambda x: x.name)

    def _load_checkpoint(self):
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        return {"current_idx": 0, "labeled": []}

    def _save_checkpoint(self):
        ckpt = {
            "current_idx": self.idx,
            "labeled": list(set(
                self.ckpt.get("labeled", []) + [str(self.img_files[self.idx])]
            ))
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(ckpt, f, indent=2)

    def _load_image(self):
        self.canvas.delete("all")
        self.boxes = []

        img_path = self.img_files[self.idx]
        try:
            self.pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"无法读取 {img_path.name}: {e}")
            return

        w, h = self.pil_img.size
        self.scale = min(DISPLAY_SIZE / max(w, h), 1.0)
        self.new_w, self.new_h = int(w * self.scale), int(h * self.scale)
        self.offset_x = (DISPLAY_SIZE - self.new_w) // 2
        self.offset_y = (DISPLAY_SIZE - self.new_h) // 2

        resized = self.pil_img.resize((self.new_w, self.new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_img)

        # 恢复已有标注
        out_stem = f"{img_path.parent.name}_{img_path.stem}"
        lbl_path = LABEL_OUT / f"{out_stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy = float(parts[1]), float(parts[2])
                    bw, bh = float(parts[3]), float(parts[4])
                    x1 = int((cx - bw / 2) * self.new_w) + self.offset_x
                    y1 = int((cy - bh / 2) * self.new_h) + self.offset_y
                    x2 = int((cx + bw / 2) * self.new_w) + self.offset_x
                    y2 = int((cy + bh / 2) * self.new_h) + self.offset_y
                    self.boxes.append((cls_id, x1, y1, x2, y2))

        self._redraw()

    def _redraw(self):
        self.canvas.delete("box")
        self.canvas.delete("tmp")
        for cls_id, x1, y1, x2, y2 in self.boxes:
            self._draw_box(cls_id, x1, y1, x2, y2)
        self._update_info()

    def _draw_box(self, cls_id, x1, y1, x2, y2):
        color = COLORS[cls_id % len(COLORS)]
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
        label = f"{cls_id}:{CLASS_NAMES[cls_id].split(':')[1][:12]}"
        self.canvas.create_rectangle(x1, y1 - 16, x1 + len(label) * 7 + 4, y1, fill=color, outline=color, tags="box")
        self.canvas.create_text(x1 + 2, y1 - 9, text=label, anchor=tk.W, fill="white", font=('', 8), tags="box")

    def _on_mouse_down(self, event):
        self.drawing = True
        self.start_x, self.start_y = event.x, event.y

    def _on_mouse_drag(self, event):
        if self.drawing:
            self.canvas.delete("tmp")
            self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline=COLORS[self.current_cls], width=1, dash=(4, 4), tags="tmp")

    def _on_mouse_up(self, event):
        self.drawing = False
        self.canvas.delete("tmp")
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        bw, bh = x2 - x1, y2 - y1
        if bw > 5 and bh > 5:
            self.boxes.append((self.current_cls, x1, y1, x2, y2))
            self._draw_box(self.current_cls, x1, y1, x2, y2)
            px_w = int(bw / self.scale)
            px_h = int(bh / self.scale)
            self.info_label.config(text=f"框: {px_w}x{px_h}px | {CLASS_NAMES[self.current_cls]}")
        self._update_info()

    def _switch_cls(self, cls_id):
        self.current_cls = cls_id
        self.cls_var.set(cls_id)

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
        self.progress_label.config(text=f"{self.idx + 1}/{len(self.img_files)}")
        self.filename_label.config(
            text=f"{img_path.parent.name}/{img_path.name}\n"
                 f"{self.pil_img.size[0]}x{self.pil_img.size[1]}px (短边{short})")
        self.box_count_label.config(text=f"框数: {len(self.boxes)}")
        labeled = len(list(LABEL_OUT.glob("*.txt")))
        self.info_label.config(text=f"已标注: {labeled}/{len(self.img_files)}")

    def _save(self):
        img_path = self.img_files[self.idx]
        out_stem = f"{img_path.parent.name}_{img_path.stem}"
        lbl_path = LABEL_OUT / f"{out_stem}.txt"

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

        # 保存图片到输出目录
        import shutil
        img_out = IMAGES_OUT / f"{out_stem}{img_path.suffix}"
        if not img_out.exists():
            img = Image.open(img_path)
            if img.mode in ('RGBA', 'P', 'LA', 'CMYK'):
                img = img.convert('RGB')
            img.save(img_out, quality=95)

        self._save_checkpoint()
        self.info_label.config(text=f"已保存 {out_stem}")

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
        labeled = len(list(LABEL_OUT.glob("*.txt")))
        messagebox.showinfo("退出", f"进度已保存。\n已标注: {labeled}/{len(self.img_files)}")
        self.root.destroy()


if __name__ == "__main__":
    LabelTool()
