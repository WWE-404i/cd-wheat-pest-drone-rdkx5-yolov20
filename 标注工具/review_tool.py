"""
快速审核工具 - V5伪标注预加载，切换自动保存，回去框还在
操作: 鼠标拖框=补框 | 数字键0-6选类 | 右键/D=删框 | N=下一张(自动保存) | Q=退出
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import json

PROJECT = Path(r"D:\python\程序\mnist-YOLO\小麦病变识别模型")
PSEUDO_DIR = PROJECT / "wheat_pseudo_v5"
SRC_DATASET = PROJECT / "wheat_disease_8class"
OUT_DIR = PROJECT / "rust_review"
CHECKPOINT = PROJECT / "review_checkpoint_v4.json"  # 锈病专项复查
REF_PANEL = PROJECT / "reference_panel.png"

TARGET_CLASSES = {0, 1, 2, 3, 4, 5, 6}  # 全部7类
MIN_TARGET_BOXES = 1
CLASS_NAMES = ["Brown_Rust","Yellow_Rust","Black_Rust","Septoria","Powdery_Mildew","Fusarium_HB","Healthy"]
COLORS = ["#FF4444","#FFAA00","#CC6600","#00CC00","#00AAAA","#0066FF","#888888"]

DISPLAY = 1000


class ReviewTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("终审 - 368张锈病(0/1/2)")
        self.root.geometry(f"{DISPLAY+300}x{DISPLAY+80}")

        self.images = []       # [(img_path_str, stem), ...]
        self.idx = 0
        self.current_img = None
        self.tk_img = None
        self.boxes = []        # [[x1,y1,x2,y2,cls],...]
        self.current_cls = 0
        self.start_x = self.start_y = None
        self.scale = 1.0
        self.offset_x = self.offset_y = 0
        self.modified = False  # 当前图是否被修改过
        self.v5_boxes_cache = {}  # stem -> [boxes] V5原始预测

        self.load_image_list()
        self.build_ui()

        self.root.bind('<Key>', self.on_key)
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Button-3>', self.delete_last_box)

        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.load_image()
        self.root.mainloop()

    def _find_source_image(self, stem):
        # 先搜原8class数据集
        for split in ["train", "val"]:
            p = SRC_DATASET / split / "images" / f"{stem}.jpg"
            if p.exists():
                return p
        # 再搜精标golden目录
        golden = PROJECT / "wheat_disease_golden"
        for split in ["train", "val"]:
            p = golden / split / "images" / f"{stem}.jpg"
            if p.exists():
                return p
        return None

    def _parse_label(self, txt_path):
        """解析label文件，返回 [{cls, xyxyn}, ...]"""
        boxes = []
        if not txt_path.exists():
            return boxes
        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1n = cx - bw / 2
            y1n = cy - bh / 2
            x2n = cx + bw / 2
            y2n = cy + bh / 2
            boxes.append({'cls': cls_id, 'xyxyn': [x1n, y1n, x2n, y2n]})
        return boxes

    def load_image_list(self):
        """第二轮复查: 只加载已手标审核的图片，聚焦叶锈vs秆锈"""
        print("扫描已审核图片...")

        done = set()
        if CHECKPOINT.exists():
            done = set(json.loads(CHECKPOINT.read_text()))

        reviewed_lbl = OUT_DIR / "labels"
        if not reviewed_lbl.exists():
            print("没有审核过的标签!")
            return

        for lbl_path in sorted(reviewed_lbl.glob("*.txt")):
            stem = lbl_path.stem
            boxes = self._parse_label(lbl_path)
            n_target = sum(1 for b in boxes if b['cls'] in TARGET_CLASSES)
            if n_target < MIN_TARGET_BOXES:
                continue
            img_path = self._find_source_image(stem)
            if img_path is None:
                continue
            self.images.append((str(img_path), stem, n_target))
            # 用V5原始预测作为"参考基线"缓存
            v5_boxes = []
            for split in ["train", "val"]:
                v5_lbl = PSEUDO_DIR / split / "labels" / f"{stem}.txt"
                if v5_lbl.exists():
                    v5_boxes = self._parse_label(v5_lbl)
                    break
            self.v5_boxes_cache[stem] = v5_boxes

        # 过滤已完成 + 按目标框数降序
        self.images = [(p, s, n) for p, s, n in self.images if s not in done]
        self.images.sort(key=lambda x: -x[2])
        print(f"待复查: {len(self.images)} 张 (已完成: {len(done)})")

    def _get_boxes(self, stem):
        """获取当前图片的框 -- 已审核用审核结果，否则用V5预测"""
        reviewed_lbl = OUT_DIR / "labels" / f"{stem}.txt"
        if reviewed_lbl.exists():
            return self._parse_label(reviewed_lbl)
        return self.v5_boxes_cache.get(stem, [])

    def build_ui(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, width=DISPLAY, height=DISPLAY, bg='#222')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = ttk.Frame(frame, width=280)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(panel, text="当前类别", font=('', 12, 'bold')).pack(pady=5)

        self.cls_btns = []
        for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
            btn = ttk.Button(panel, text=f"{i}: {name}",
                             command=lambda c=i: self.set_class(c))
            btn.pack(fill=tk.X, pady=2)
            self.cls_btns.append(btn)
        self.update_cls_btn()

        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=10)

        self.info_label = ttk.Label(panel, text="", font=('', 10))
        self.info_label.pack(pady=5)

        self.progress_label = ttk.Label(panel, text="", font=('', 9))
        self.progress_label.pack(pady=5)

        self.modified_label = ttk.Label(panel, text="", font=('', 9))
        self.modified_label.pack(pady=2)

        ttk.Button(panel, text="下一张 (N) 自动保存", command=self.next_image).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="删最后框 (D/右键)", command=self.delete_last_box).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="清空所有框 (C)", command=self.clear_boxes).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="手动保存 (S)", command=self.save).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="前一张 (Left)", command=self.prev_image).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="撤销修改 恢复V5 (R)", command=self.reset_to_v5).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="参考对照图 (F1)", command=self.show_reference).pack(fill=tk.X, pady=2)

        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- 参考鉴别面板（始终可见）---
        ref_frame = ttk.LabelFrame(panel, text="鉴别参考")
        ref_frame.pack(fill=tk.X, pady=5)

        ref_text = (
            "★ 重点区分 ★\n"
            "0 Brown_Rust 叶锈:\n"
            "  橙色小圆斑(1-2mm)，随机散落\n"
            "  不成条，不撕裂表皮\n"
            "  只在叶片上，不上茎秆\n"
            "  斑点多而小，像撒了橙粉\n\n"
            "2 Black_Rust 秆锈:\n"
            "  暗红褐色大斑，表皮明显撕裂\n"
            "  茎秆和叶鞘上也有！\n"
            "  斑块大而不规则，边缘破碎\n"
            "  颜色更深(暗红/棕黑)\n\n"
            "----------\n"
            "1 Yellow_Rust 条锈:\n"
            "  亮黄花序，沿叶脉成条状\n\n"
            "3 Septoria 叶枯:\n"
            "  褐斑+黄晕+小黑点(分生孢子器)"
        )
        self.ref_label = ttk.Label(ref_frame, text=ref_text,
                                   font=('', 8), justify=tk.LEFT)
        self.ref_label.pack(padx=5, pady=5)

        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(panel, text="", font=('', 9, 'italic'))
        self.status_label.pack(pady=5)

        ttk.Label(panel, text="操作: 拖拽画框 | 0-6选类\n右键删框 | N下一张(自动保存) | Q退出",
                  font=('', 8)).pack(pady=10)

    def set_class(self, c):
        self.current_cls = c
        self.update_cls_btn()

    def update_cls_btn(self):
        for i, btn in enumerate(self.cls_btns):
            if i == self.current_cls:
                btn.state(['pressed'])
            else:
                btn.state(['!pressed'])

    def load_image(self):
        if not self.images:
            self.canvas.delete("all")
            self.canvas.create_text(DISPLAY//2, DISPLAY//2, text="全部完成!",
                                    fill="white", font=('', 24))
            self.modified_label.config(text="")
            return

        self.canvas.delete("all")
        img_path_str, stem, _ = self.images[self.idx]
        img_path = Path(img_path_str)
        pil_img = Image.open(img_path)
        w, h = pil_img.size

        self.scale = min(DISPLAY / w, DISPLAY / h)
        self.offset_x = (DISPLAY - w * self.scale) // 2
        self.offset_y = (DISPLAY - h * self.scale) // 2

        self.current_img = pil_img.resize(
            (int(w * self.scale), int(h * self.scale)), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.current_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW,
                                 image=self.tk_img)

        # 加载框：优先已审核的，否则V5预测
        all_boxes = self._get_boxes(stem)
        self.boxes = []
        for p in all_boxes:
            x1n, y1n, x2n, y2n = p['xyxyn']
            x1 = x1n * w * self.scale + self.offset_x
            y1 = y1n * h * self.scale + self.offset_y
            x2 = x2n * w * self.scale + self.offset_x
            y2 = y2n * h * self.scale + self.offset_y
            self.boxes.append([x1, y1, x2, y2, p['cls']])

        self.modified = False
        self.draw_boxes()

        n_brown = sum(1 for b in self.boxes if b[4] == 0)
        n_black = sum(1 for b in self.boxes if b[4] == 2)
        n_total = len(self.boxes)
        self.info_label.config(
            text=f"{img_path.name}\n总框: {n_total} | Brown_Rust: {n_brown}  Black_Rust: {n_black}\n来源: 手标审核")
        self.progress_label.config(text=f"{self.idx+1} / {len(self.images)}")
        self.root.title(f"审核 [{self.idx+1}/{len(self.images)}] {img_path.name}")

    def draw_boxes(self):
        self.canvas.delete("box")
        class_counts = {}
        for b in self.boxes:
            x1, y1, x2, y2, cls = b
            color = COLORS[cls]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color,
                                         width=2, tags="box")
            name = CLASS_NAMES[cls]
            self.canvas.create_text(x1+4, y1+4, text=f"{cls}:{name}",
                                    anchor=tk.NW, fill=color,
                                    font=('', 9, 'bold'), tags="box")
            class_counts[cls] = class_counts.get(cls, 0) + 1

        status = " | ".join(
            f"{CLASS_NAMES[c]}:{n}" for c, n in sorted(class_counts.items()))
        self.status_label.config(text=status)
        self.modified_label.config(
            text="[已修改]" if self.modified else "")

    def _mark_modified(self):
        if not self.modified:
            self.modified = True
            self.modified_label.config(text="[已修改]")

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        if self.start_x is None:
            return
        self.canvas.delete("preview")
        self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline=COLORS[self.current_cls], width=2, dash=(5,5), tags="preview")

    def on_mouse_up(self, event):
        if self.start_x is None:
            return
        self.canvas.delete("preview")
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        if x2 - x1 > 5 and y2 - y1 > 5:
            self.boxes.append([x1, y1, x2, y2, self.current_cls])
            self._mark_modified()
            self.draw_boxes()
        self.start_x = None

    def delete_last_box(self, event=None):
        if self.boxes:
            self.boxes.pop()
            self._mark_modified()
            self.draw_boxes()

    def clear_boxes(self):
        if self.boxes:
            self.boxes = []
            self._mark_modified()
            self.draw_boxes()

    def reset_to_v5(self):
        """撤销修改，恢复为V5原始预测"""
        if not self.images:
            return
        stem = self.images[self.idx][1]
        v5_boxes = self.v5_boxes_cache.get(stem, [])
        # 删除已审核文件
        reviewed_lbl = OUT_DIR / "labels" / f"{stem}.txt"
        if reviewed_lbl.exists():
            reviewed_lbl.unlink()
        # 重新加载
        img_path = Path(self.images[self.idx][0])
        pil_img = Image.open(img_path)
        w, h = pil_img.size
        self.boxes = []
        for p in v5_boxes:
            x1n, y1n, x2n, y2n = p['xyxyn']
            x1 = x1n * w * self.scale + self.offset_x
            y1 = y1n * h * self.scale + self.offset_y
            x2 = x2n * w * self.scale + self.offset_x
            y2 = y2n * h * self.scale + self.offset_y
            self.boxes.append([x1, y1, x2, y2, p['cls']])
        self.modified = False
        self.draw_boxes()
        self.status_label.config(text="已恢复V5预测")

    def on_key(self, event):
        c = event.char
        if c in '0123456':
            self.set_class(int(c))
        elif c.lower() == 'n' or event.keysym == 'space':
            self.next_image()
        elif c.lower() == 'd':
            self.delete_last_box()
        elif c.lower() == 'c':
            self.clear_boxes()
        elif c.lower() == 's':
            self.save()
        elif c.lower() == 'r':
            self.reset_to_v5()
        elif event.keysym == 'Left':
            self.prev_image()
        elif event.keysym == 'Right':
            self.next_image()
        elif c.lower() == 'q':
            self.on_quit()
        elif event.keysym == 'F1':
            self.show_reference()

    def next_image(self):
        if not self.images:
            return
        self.save()  # 自动保存当前
        self.idx = (self.idx + 1) % len(self.images)
        self.load_image()

    def prev_image(self):
        if not self.images:
            return
        self.save()  # 自动保存当前
        self.idx = (self.idx - 1) % len(self.images)
        self.load_image()

    def save(self):
        """保存当前标注（只存label文件，不复制原图省空间）"""
        if not self.images:
            return
        img_path = Path(self.images[self.idx][0])
        stem = self.images[self.idx][1]

        out_lbl = OUT_DIR / "labels"
        out_lbl.mkdir(parents=True, exist_ok=True)

        # 保存标签
        pil_img = Image.open(img_path)
        w, h = pil_img.size
        lines = []
        for x1, y1, x2, y2, cls in self.boxes:
            cx = ((x1 - self.offset_x) / self.scale +
                  (x2 - self.offset_x) / self.scale) / 2 / w
            cy = ((y1 - self.offset_y) / self.scale +
                  (y2 - self.offset_y) / self.scale) / 2 / h
            bw = abs((x2 - x1) / self.scale) / w
            bh = abs((y2 - y1) / self.scale) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_lbl / f"{stem}.txt").write_text("\n".join(lines))

        # 更新断点
        done = []
        if CHECKPOINT.exists():
            done = json.loads(CHECKPOINT.read_text())
        if stem not in done:
            done.append(stem)
        CHECKPOINT.write_text(json.dumps(done, ensure_ascii=False))

        if self.modified:
            self.modified = False
            self.modified_label.config(text="")
            self.status_label.config(text=f"已保存: {img_path.name}")
            self.root.after(1200, lambda: self.status_label.config(text=""))

    def show_reference(self):
        """弹出参考对照图窗口"""
        if not REF_PANEL.exists():
            self.status_label.config(text="参考图未生成，先运行 create_reference_panel.py")
            return

        ref_win = tk.Toplevel(self.root)
        ref_win.title("四类病变参考对照")
        ref_win.resizable(True, True)

        pil_img = Image.open(REF_PANEL)
        self.ref_tk = ImageTk.PhotoImage(pil_img)

        canvas = tk.Canvas(ref_win, width=pil_img.width, height=pil_img.height)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 滚动条
        hbar = ttk.Scrollbar(ref_win, orient=tk.HORIZONTAL, command=canvas.xview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = ttk.Scrollbar(ref_win, orient=tk.VERTICAL, command=canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        canvas.create_image(0, 0, anchor=tk.NW, image=self.ref_tk)
        canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))

    def on_quit(self):
        self.save()  # 退出前自动保存当前
        done = []
        if CHECKPOINT.exists():
            done = json.loads(CHECKPOINT.read_text())
        CHECKPOINT.write_text(json.dumps(done, ensure_ascii=False))
        print(f"断点已保存: {len(done)} 张已审核")
        self.root.destroy()


if __name__ == "__main__":
    ReviewTool()
