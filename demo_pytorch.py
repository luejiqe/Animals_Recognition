import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

# 定义模型(与训练时相同)
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        from torchvision.models import EfficientNet_B6_Weights
        self.base_model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)

        # 冻结底层参数
        for i, param in enumerate(self.base_model.parameters()):
            if i < 200:
                param.requires_grad = False

        # 获取最后一层的输入特征数
        in_features = self.base_model.classifier[1].in_features

        # 替换分类器
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class AnimalRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("动物识别系统")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f4f8")

        # 配置参数
        self.model_path = './pytorch_model.pt'
        self.class_names_path = 'class.txt'
        self.img_size = (456, 456)
        self.num_classes = 100

        # 设备配置
        self.use_gpu = True
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("使用 CPU 进行推理")

        # 颜色配置
        self.colors = {
            'primary': '#3b82f6',
            'primary_dark': '#1e40af',
            'primary_light': '#93c5fd',
            'success': '#10b981',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'text': '#1f2937',
            'text_light': '#6b7280',
            'border': '#e5e7eb',
            'bg_card': '#ffffff',
            'bg_main': '#f0f4f8'
        }

        # 初始化变量
        self.model = None
        self.class_names = []
        self.current_image_path = None
        self.current_image = None

        # 加载模型和类别
        self.load_model()
        self.load_class_names()

        # 设置样式
        self.setup_styles()

        # 创建UI
        self.create_ui()

    def setup_styles(self):
        """设置ttk样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 主背景样式
        style.configure("Main.TFrame", background=self.colors['bg_main'])
        style.configure("Card.TFrame", background=self.colors['bg_card'], relief=tk.FLAT)
        style.configure("White.TFrame", background=self.colors['bg_card'])

        # 标签样式
        style.configure("White.TLabel", background=self.colors['bg_card'], foreground=self.colors['text'])
        style.configure("Title.TLabel", background=self.colors['bg_card'],
                       foreground=self.colors['primary_dark'], font=("Segoe UI", 20, "bold"))

        # 按钮样式
        style.configure("Normal.TButton", font=("Segoe UI", 11), padding=10,
                       background=self.colors['primary'], foreground="white")
        style.map("Normal.TButton",
                 background=[('active', self.colors['primary_dark']), ('pressed', self.colors['primary_dark'])])

        style.configure("Normal.Hover.TButton", font=("Segoe UI", 11), padding=10,
                       background=self.colors['primary_dark'], foreground="white")

        style.configure("Accent.TButton", font=("Segoe UI", 11), padding=10,
                       background=self.colors['success'], foreground="white")
        style.map("Accent.TButton",
                 background=[('active', '#059669'), ('pressed', '#059669'), ('disabled', '#9ca3af')])

        style.configure("Accent.Hover.TButton", font=("Segoe UI", 11), padding=10,
                       background='#059669', foreground="white")

    def create_ui(self):
        """创建主界面"""
        # 主容器
        main_container = ttk.Frame(self.root, style="Main.TFrame", padding=30)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_frame = ttk.Frame(main_container, style="Main.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = ttk.Label(title_frame, text="🐾 动物识别系统",
                               font=("Segoe UI", 24, "bold"),
                               foreground=self.colors['primary_dark'],
                               background=self.colors['bg_main'])
        title_label.pack()

        subtitle_label = ttk.Label(title_frame, text="基于 EfficientNet-B6 深度学习模型",
                                  font=("Segoe UI", 11),
                                  foreground=self.colors['text_light'],
                                  background=self.colors['bg_main'])
        subtitle_label.pack()

        # 内容区域
        content_frame = ttk.Frame(main_container, style="Main.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧图片区域
        left_frame = ttk.Frame(content_frame, style="Main.TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        # 图片上传卡片
        upload_frame = ttk.Frame(left_frame, style="Card.TFrame", padding=20)
        upload_frame.pack(fill=tk.BOTH, expand=True)

        # 标题区域
        title_container = ttk.Frame(upload_frame, style="White.TFrame")
        title_container.pack(fill=tk.X, pady=(0, 10))

        upload_title = ttk.Label(title_container, text="🖼️ 图片上传识别",
                                font=("Segoe UI", 16, "bold"), style="White.TLabel")
        upload_title.pack(side=tk.LEFT)

        # 图片显示区域
        image_display_frame = ttk.Frame(upload_frame, style="Card.TFrame")
        image_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 创建Canvas显示图片
        self.image_canvas = tk.Canvas(image_display_frame, bg="#f8fafc",
                                      highlightthickness=1,
                                      highlightbackground=self.colors['border'])
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        # 图片标签
        self.recognition_image_label = ttk.Label(
            self.image_canvas,
            text="📸 请上传动物图片进行识别\n\n支持格式: JPG, PNG, BMP\n建议尺寸: 456×456 或更大",
            anchor=tk.CENTER,
            font=("Segoe UI", 12),
            style="White.TLabel",
            justify=tk.CENTER
        )
        self.image_canvas.create_window(0, 0, anchor=tk.NW, window=self.recognition_image_label)

        # Canvas居中
        def center_image(event):
            canvas_width = event.width
            canvas_height = event.height
            self.image_canvas.coords(self.image_canvas.find_all()[0],
                                    canvas_width//2, canvas_height//2)
            self.image_canvas.itemconfig(self.image_canvas.find_all()[0], anchor=tk.CENTER)

        self.image_canvas.bind('<Configure>', center_image)

        # 按钮区域
        button_separator = ttk.Separator(upload_frame, orient='horizontal')
        button_separator.pack(fill=tk.X, pady=(15, 10))

        button_frame = ttk.Frame(upload_frame, style="White.TFrame")
        button_frame.pack(fill=tk.X, pady=(0, 5))

        button_inner = ttk.Frame(button_frame, style="White.TFrame")
        button_inner.pack(expand=True)

        # 上传按钮
        upload_btn = ttk.Button(button_inner, text="📁 上传图片",
                               command=self.upload_image,
                               style="Normal.TButton", width=18)
        upload_btn.pack(side=tk.LEFT, padx=8, pady=5)
        upload_btn.bind("<Enter>", lambda e, b=upload_btn: b.config(style="Normal.Hover.TButton"))
        upload_btn.bind("<Leave>", lambda e, b=upload_btn: b.config(style="Normal.TButton"))

        # 识别按钮
        self.start_recognition_btn = ttk.Button(button_inner, text="🔍 开始识别",
                                               command=self.start_recognition,
                                               state=tk.DISABLED,
                                               style="Accent.TButton", width=18)
        self.start_recognition_btn.pack(side=tk.LEFT, padx=8, pady=5)
        self.start_recognition_btn.bind("<Enter>", lambda e, b=self.start_recognition_btn: b.config(style="Accent.Hover.TButton"))
        self.start_recognition_btn.bind("<Leave>", lambda e, b=self.start_recognition_btn: b.config(style="Accent.TButton"))

        # 提示信息
        hint_label = ttk.Label(upload_frame,
                              text="💡 提示：上传图片后点击\"开始识别\"按钮进行AI识别",
                              font=("Segoe UI", 9),
                              foreground=self.colors['text_light'],
                              style="White.TLabel")
        hint_label.pack(pady=(5, 0))

        # 右侧结果区域
        right_frame = ttk.Frame(content_frame, style="Main.TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 结果卡片
        result_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=20)
        result_frame.pack(fill=tk.BOTH, expand=True)

        # 标题区域
        result_title_container = ttk.Frame(result_frame, style="White.TFrame")
        result_title_container.pack(fill=tk.X, pady=(0, 10))

        result_title = ttk.Label(result_title_container, text="🎯 识别结果",
                                font=("Segoe UI", 16, "bold"), style="White.TLabel")
        result_title.pack(side=tk.LEFT)

        # 清空按钮
        clear_btn = ttk.Button(result_title_container, text="🗑️ 清空",
                              command=lambda: self.result_text.delete(1.0, tk.END),
                              style="Normal.TButton", width=10)
        clear_btn.pack(side=tk.RIGHT)
        clear_btn.bind("<Enter>", lambda e, b=clear_btn: b.config(style="Normal.Hover.TButton"))
        clear_btn.bind("<Leave>", lambda e, b=clear_btn: b.config(style="Normal.TButton"))

        # 分隔线
        ttk.Separator(result_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))

        # 结果文本框
        result_container = ttk.Frame(result_frame, style="White.TFrame")
        result_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(result_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(result_container, height=15, width=50, yscrollcommand=scrollbar.set,
                                  font=("Segoe UI", 11), wrap=tk.WORD, padx=15, pady=15,
                                  relief=tk.FLAT, borderwidth=0, background="#ffffff",
                                  highlightthickness=1, highlightbackground=self.colors['border'],
                                  highlightcolor=self.colors['primary'])
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)

        # 文本样式
        self.result_text.tag_configure("title", font=("Segoe UI", 14, "bold"),
                                      foreground=self.colors['primary_dark'],
                                      spacing1=10, spacing3=10)
        self.result_text.tag_configure("result", font=("Segoe UI", 12),
                                      foreground=self.colors['text'],
                                      spacing1=5)
        self.result_text.tag_configure("highlight", font=("Segoe UI", 13, "bold"),
                                      foreground=self.colors['danger'],
                                      spacing1=8, spacing3=8)
        self.result_text.tag_configure("info", font=("Segoe UI", 10),
                                      foreground=self.colors['text_light'],
                                      spacing1=3)

        # 进度条
        self.progress_frame = ttk.Frame(result_frame, style="White.TFrame")
        self.progress_frame.pack(fill=tk.X, pady=(15, 0))

        self.progress_label = ttk.Label(self.progress_frame, text="正在识别中...",
                                       font=("Segoe UI", 10),
                                       foreground=self.colors['primary'],
                                       style="White.TLabel")
        self.progress_label.pack(pady=(0, 5))

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=300)
        self.progress_bar.pack(fill=tk.X)

        self.progress_frame.pack_forget()

    def load_model(self):
        """加载PyTorch模型"""
        try:
            self.model = EfficientNetClassifier(num_classes=self.num_classes)

            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            print("模型加载成功！")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            print(f"模型加载失败: {str(e)}")

    def load_class_names(self):
        """加载类别名称"""
        try:
            with open(self.class_names_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"成功加载 {len(self.class_names)} 个类别")
        except Exception as e:
            messagebox.showerror("错误", f"类别文件加载失败: {str(e)}")
            print(f"类别文件加载失败: {str(e)}")

    def upload_image(self):
        """上传图片"""
        file_path = filedialog.askopenfilename(
            title="选择动物图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if not file_path:
            return

        try:
            # 加载图片
            img = Image.open(file_path)
            self.current_image_path = file_path
            self.current_image = img

            # 显示图片
            display_size = (400, 400)
            img_display = img.copy()
            img_display.thumbnail(display_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img_display)
            self.recognition_image_label.configure(image=photo, text="")
            self.recognition_image_label.image = photo

            # 启用识别按钮
            self.start_recognition_btn.config(state=tk.NORMAL)

            # 清空之前的结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "✅ 图片上传成功！\n\n", "title")
            self.result_text.insert(tk.END, f"文件路径: {file_path}\n", "info")
            self.result_text.insert(tk.END, f"图片尺寸: {img.size[0]} × {img.size[1]}\n", "info")
            self.result_text.insert(tk.END, "\n点击\"开始识别\"按钮进行AI识别", "result")

        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败: {str(e)}")

    def start_recognition(self):
        """开始识别"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先上传图片！")
            return

        # 显示进度条
        self.progress_frame.pack(fill=tk.X, pady=(15, 0))
        self.progress_bar.start(10)

        # 禁用按钮
        self.start_recognition_btn.config(state=tk.DISABLED)

        # 在新线程中执行识别
        thread = threading.Thread(target=self.perform_recognition)
        thread.daemon = True
        thread.start()

    def perform_recognition(self):
        """执行识别"""
        try:
            # 图片预处理
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            img_tensor = transform(self.current_image).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top5_prob, top5_idx = torch.topk(probabilities, 5)

            # 构建结果
            result_str = "🎯 识别结果 Top-5:\n\n"
            for i in range(5):
                class_idx = top5_idx[0][i].item()
                prob = top5_prob[0][i].item()
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"类别{class_idx}"
                result_str += f"{i+1}. {class_name}\n"
                result_str += f"   置信度: {prob*100:.2f}%\n\n"

            # 在主线程中更新UI
            self.root.after(0, self.show_recognition_result, result_str)

        except Exception as e:
            error_msg = f"识别失败: {str(e)}"
            self.root.after(0, self.show_recognition_error, error_msg)

    def show_recognition_result(self, result_str):
        """显示识别结果"""
        # 隐藏进度条
        self.progress_bar.stop()
        self.progress_frame.pack_forget()

        # 启用按钮
        self.start_recognition_btn.config(state=tk.NORMAL)

        # 显示结果
        self.result_text.delete(1.0, tk.END)

        lines = result_str.split('\n')
        for line in lines:
            if line.startswith('🎯'):
                self.result_text.insert(tk.END, line + '\n', "title")
            elif line.strip() and line[0].isdigit():
                self.result_text.insert(tk.END, line + '\n', "highlight")
            elif '置信度' in line:
                self.result_text.insert(tk.END, line + '\n', "result")
            else:
                self.result_text.insert(tk.END, line + '\n', "result")

    def show_recognition_error(self, error_msg):
        """显示识别错误"""
        # 隐藏进度条
        self.progress_bar.stop()
        self.progress_frame.pack_forget()

        # 启用按钮
        self.start_recognition_btn.config(state=tk.NORMAL)

        # 显示错误
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "❌ 识别失败\n\n", "title")
        self.result_text.insert(tk.END, error_msg, "result")

        messagebox.showerror("错误", error_msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalRecognitionApp(root)
    root.mainloop()
