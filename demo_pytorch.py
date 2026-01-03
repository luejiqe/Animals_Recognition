import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk, ImageOps, ImageFilter, ImageDraw, ImageFont, ImageEnhance
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import json
import threading
import time
import math
from datetime import datetime
import glob

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
        self.root.title("动物世界探索家")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f0f4f8")  # 柔和的浅蓝灰色背景

        # 设置应用图标
        try:
            self.root.iconbitmap("animal_icon.ico")  # 动物图标
        except:
            pass

        # 配置参数
        self.model_path = './pytorch_model.pt'  # PyTorch 模型权重路径
        self.image_dir = 'test'
        self.class_names_path = 'class.txt'
        self.img_size = (456, 456)
        self.num_classes = 100
        self.zoo_icons_dir = 'zoo_icons'  # 动物图标目录
        self.animal_images_dir = 'Animal'  # 动物图片目录
        self.bg_patterns_dir = 'bg_patterns'  # 背景图案目录

        # 设备配置：优先使用 GPU，如果不可用则回退到 CPU
        self.use_gpu = True  # 可以设置为 False 强制使用 CPU
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"GUI 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("GUI 使用 CPU 进行推理")

        # 动物分类
        self.land_animals = [
            'antelope', 'badger', 'bear', 'bison', 'boar', 'camel', 'capybara', 'cat',
            'cow', 'coyote', 'crocodile', 'chimpanzee', 'deer', 'dog', 'donkey', 'elephant', 'fox',
            'giraffe', 'goat', 'gorilla', 'hamster', 'hare', 'hedgehog', 'hippopotamus',
            'horse', 'hyena', 'kangaroo', 'koala', 'leopard', 'lion', 'mandrill',
            'mouse', 'orangutan', 'otter', 'panda', 'pangolin', 'pig', 'platypus',
            'porcupine', 'possum', 'raccoon', 'reindeer', 'rhinoceros', 'sheep',
            'squirrel', 'tiger', 'walrus', 'wombat', 'zebra'
        ]

        self.sea_animals = [
            'crab', 'dolphin', 'jellyfish', 'lobster', 'octopus', 'penguin',
            'pufferfish', 'scallop', 'seahorse', 'seal', 'shark', 'squid',
            'starfish', 'turtle', 'whale'
        ]

        self.air_animals = [
            'bat', 'bee', 'beetle', 'butterfly', 'cockroach', 'caterpillar',
            'crow', 'dragonfly', 'duck', 'eagle', 'flamingo', 'fly', 'frog',
            'goldfish', 'goose', 'grasshopper', 'hornbill', 'hummingbird',
            'ladybugs', 'lizard', 'mantis', 'mosquito', 'moth', 'owl',
            'parrot', 'peacock', 'pelecaniformes', 'pigeon', 'sandpiper',
            'scorpion', 'snake', 'sparrow', 'spider', 'swan', 'turkey',
            'woodpecker'
        ]

        # 当前上传的图片路径
        self.current_image_path = None
        self.processed_img = None  # 处理后的图片

        # 用户统计
        self.user_stats = {
            "total_recognitions": 0,
            "correct_guesses": 0,
            "animals_unlocked": 0,
            "last_played": None
        }
        self.load_user_stats()

        # 创建样式
        self.setup_styles()

        # 创建主框架
        self.main_frame = ttk.Frame(self.root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 底部状态栏 - 先创建状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W, style="Status.TLabel")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 已解锁的动物集合 - 现在再加载已解锁动物
        self.unlocked_animals = set()
        self.load_unlocked_animals()  # 加载已解锁动物

        # 加载模型
        self.model = None
        self.model_loaded = False  # 模型是否已加载的标志
        self.transform = None  # PyTorch 图像预处理
        self.load_model()

        # 加载类别名称
        self.class_names = []
        self.load_class_names()

        # 创建标题和动画效果
        self.create_main_ui()

        # 更新状态栏
        self.update_status("应用已启动，欢迎使用动物世界探索家！")

        # 启动背景动画
        self.animate_particles()

    def setup_styles(self):
        """设置应用样式 - 采用更现代的设计风格"""
        style = ttk.Style()

        # 配置主题
        style.theme_use('clam')

        # 主色调方案：使用自然的蓝绿色调，代表自然和动物世界
        self.colors = {
            'primary': '#2a9d8f',       # 主色调：自然绿蓝色
            'primary_light': '#2ecc71', # 亮色调：浅绿色
            'primary_dark': '#264653',  # 暗色调：深青蓝色
            'secondary': '#e9c46a',     # 辅助色：暖黄色
            'accent': '#f4a261',        # 强调色：橙色
            'danger': '#e76f51',        # 危险色：红色
            'background': '#f0f4f8',    # 背景色
            'card': '#ffffff',          # 卡片背景
            'text': '#264653',          # 文本色
            'text_light': '#64748b',    # 次要文本色
            'border': '#e2e8f0',        # 边框色
            'shadow': '#d1d5db'         # 阴影色
        }

        # 配置主框架样式
        style.configure("Main.TFrame", background=self.colors['background'])

        # 配置阴影样式
        style.configure("Shadow.TFrame", background=self.colors['shadow'])

        # 配置标题样式
        style.configure("Title.TLabel",
                       background=self.colors['background'],
                       foreground=self.colors['primary_dark'],
                       font=("Segoe UI", 28, "bold"))
        style.configure("Subtitle.TLabel",
                       background=self.colors['background'],
                       foreground=self.colors['text_light'],
                       font=("Segoe UI", 14))

        # 配置按钮框架样式
        style.configure("ButtonFrame.TFrame", background=self.colors['background'])

        # 配置强调按钮样式 - 增加圆角效果和阴影
        style.configure("Accent.TButton",
                       font=("Times New Roman", 16, "bold"),
                       padding=12,
                       background=self.colors['primary'],
                       foreground="white",
                       borderwidth=0,
                       focuscolor="none")
        style.map("Accent.TButton",
                 background=[('active', '#21867a'), ('pressed', '#1e7e6f')])

        # 配置强调按钮悬停样式
        style.configure("Accent.Hover.TButton",
                       font=("Times New Roman", 16, "bold"),
                       padding=12,
                       background=self.colors['primary_light'],
                       foreground="white",
                       borderwidth=0,
                       focuscolor="none")

        # 配置普通按钮样式
        style.configure("Normal.TButton",
                       font=("Segoe UI", 10),
                       padding=8,
                       background=self.colors['card'],
                       foreground=self.colors['text'],
                       borderwidth=1,
                       bordercolor=self.colors['border'])
        style.map("Normal.TButton",
                 background=[('active', '#f1f5f9'), ('pressed', '#e2e8f0')])

        # 配置普通按钮悬停样式
        style.configure("Normal.Hover.TButton",
                       font=("Segoe UI", 10),
                       padding=8,
                       background="#f1f5f9",
                       foreground=self.colors['text'],
                       borderwidth=1,
                       bordercolor=self.colors['border'])

        # 配置进度条样式
        style.configure("TProgressbar", thickness=8, background=self.colors['primary'])

        # 配置状态栏样式
        style.configure("Status.TLabel",
                       background=self.colors['primary_dark'],
                       foreground="white",
                       padding=5,
                       font=("Segoe UI", 9))

        # 配置卡片样式 - 增加阴影效果
        style.configure("Card.TFrame",
                       background=self.colors['card'],
                       relief=tk.RAISED,
                       borderwidth=1,
                       bordercolor=self.colors['border'])

        # 配置卡片悬停样式
        style.configure("Hover.TFrame",
                       background=self.colors['card'],
                       relief=tk.RAISED,
                       borderwidth=2,
                       bordercolor=self.colors['primary'])

        # 配置白色背景样式
        style.configure("White.TFrame", background=self.colors['card'])
        style.configure("White.TLabel", background=self.colors['card'])

        # 配置选项卡样式
        style.configure("Custom.TNotebook",
                       background=self.colors['background'],
                       borderwidth=0)
        style.configure("Custom.TNotebook.Tab",
                       background="#e2e8f0",
                       padding=[15, 8],
                       font=("Segoe UI", 10, "bold"),
                       borderwidth=0)
        style.map("Custom.TNotebook.Tab",
                 background=[("selected", self.colors['primary']), ("active", "#cbd5e1")],
                 foreground=[("selected", "white"), ("active", self.colors['text'])])

    def create_main_ui(self):
        """创建主页面UI - 更具视觉吸引力的设计"""
        # 添加背景装饰
        self.add_background_decorations()

        # 创建标题容器，增加视觉层次感
        title_container = ttk.Frame(self.main_frame, style="Main.TFrame")
        title_container.pack(pady=30)

        # 创建标题，添加微妙的阴影效果
        title_label = ttk.Label(title_container, text="动物世界探索家",
                               font=("Segoe UI", 32, "bold"), style="Title.TLabel")
        title_label.pack()

        # 添加标题下方的装饰线
        separator = ttk.Separator(title_container, orient="horizontal")
        separator.pack(fill=tk.X, padx=150, pady=10)

        # 创建副标题
        subtitle_label = ttk.Label(self.main_frame, text="探索、识别、学习动物世界的奇妙",
                                  font=("Segoe UI", 14), style="Subtitle.TLabel")
        subtitle_label.pack(pady=(0, 30))

        # 创建统计信息卡片，带有轻微的阴影效果
        stats_frame = ttk.Frame(self.main_frame, style="Card.TFrame", padding=15)
        stats_frame.pack(pady=20, padx=100, fill=tk.X)

        # 添加卡片阴影效果（通过框架嵌套实现）
        stats_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        stats_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        stats_text = f"已解锁动物: {len(self.unlocked_animals)}/{len(self.class_names)} | " \
                    f"识别次数: {self.user_stats['total_recognitions']} | " \
                    f"游戏得分: {self.user_stats['correct_guesses']}"

        stats_label = ttk.Label(stats_frame, text=stats_text, font=("Segoe UI", 11), background=self.colors['card'])
        stats_label.pack()

        # 创建功能选择按钮区域，使用更现代的卡片布局
        buttons_card = ttk.Frame(self.main_frame, style="Card.TFrame", padding=30)
        buttons_card.pack(pady=30, padx=100, fill=tk.X)

        # 按钮卡片阴影
        buttons_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        buttons_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        # 按钮容器，使按钮居中
        button_frame = ttk.Frame(buttons_card, style="ButtonFrame.TFrame")
        button_frame.pack()

        # 使用更美观的按钮，增加图标和悬停效果
        button_style = {"width": 25, "padding": (15, 10)}

        btn1 = ttk.Button(button_frame, text="🐾 动物识别", command=self.show_animal_recognition,
                  style="Accent.TButton", **button_style)
        btn1.pack(pady=15)
        btn1.bind("<Enter>", lambda e, b=btn1: b.config(style="Accent.Hover.TButton"))
        btn1.bind("<Leave>", lambda e, b=btn1: b.config(style="Accent.TButton"))

        btn2 = ttk.Button(button_frame, text="🎮 动物认识小游戏", command=self.show_animal_game,
                  style="Accent.TButton", **button_style)
        btn2.pack(pady=15)
        btn2.bind("<Enter>", lambda e, b=btn2: b.config(style="Accent.Hover.TButton"))
        btn2.bind("<Leave>", lambda e, b=btn2: b.config(style="Accent.TButton"))

        btn3 = ttk.Button(button_frame, text="🏞️ 动物园图鉴", command=self.show_virtual_zoo,
                  style="Accent.TButton", **button_style)
        btn3.pack(pady=15)
        btn3.bind("<Enter>", lambda e, b=btn3: b.config(style="Accent.Hover.TButton"))
        btn3.bind("<Leave>", lambda e, b=btn3: b.config(style="Accent.TButton"))

        # 初始化各个功能页面
        self.recognition_frame = None
        self.game_frame = None
        self.zoo_frame = None

    def add_background_decorations(self):
        """添加背景装饰元素，增强视觉效果"""
        # 尝试加载背景图案
        try:
            if os.path.exists(self.bg_patterns_dir):
                pattern_files = [f for f in os.listdir(self.bg_patterns_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if pattern_files:
                    # 左侧装饰
                    pattern_path = os.path.join(self.bg_patterns_dir, random.choice(pattern_files))
                    img = Image.open(pattern_path).convert("RGBA")
                    img = img.resize((150, 150), Image.LANCZOS)
                    # 降低透明度
                    alpha = img.split()[3]
                    alpha = ImageEnhance.Brightness(alpha).enhance(0.1)
                    img.putalpha(alpha)
                    photo = ImageTk.PhotoImage(img)

                    left_decor = ttk.Label(self.main_frame, image=photo, style="Main.TFrame")
                    left_decor.image = photo
                    left_decor.place(x=10, y=10)

                    # 右侧装饰
                    pattern_path = os.path.join(self.bg_patterns_dir, random.choice(pattern_files))
                    img = Image.open(pattern_path).convert("RGBA")
                    img = img.resize((150, 150), Image.LANCZOS)
                    alpha = img.split()[3]
                    alpha = ImageEnhance.Brightness(alpha).enhance(0.1)
                    img.putalpha(alpha)
                    photo = ImageTk.PhotoImage(img)

                    right_decor = ttk.Label(self.main_frame, image=photo, style="Main.TFrame")
                    right_decor.image = photo
                    right_decor.place(x=900, y=600)
        except Exception as e:
            print(f"加载背景装饰失败: {e}")
            # 即使失败也继续执行，不影响主功能

    def float_animation(self, label, x, y, speed, drift):
        """浮动动画效果 - 更自然的轨迹"""
        if y > -50 and label.winfo_exists():
            y -= speed  # 上移速度
            x += drift  # 水平漂移
            label.place(x=x, y=y)
            self.root.after(30, lambda: self.float_animation(label, x, y, speed, drift))
        else:
            if label.winfo_exists():
                # 淡出效果
                self.fade_out(label)

    def fade_out(self, widget):
        """控件淡出效果"""
        if widget.winfo_exists():
            try:
                # 获取当前透明度
                alpha = widget.attributes("-alpha")
                if alpha > 0:
                    widget.attributes("-alpha", alpha - 0.1)
                    self.root.after(30, lambda: self.fade_out(widget))
                else:
                    widget.destroy()
            except:
                # 某些平台可能不支持透明度
                widget.destroy()

    def animate_particles(self):
        """添加背景粒子动画，增强深度感"""
        if hasattr(self, 'main_frame') and self.main_frame.winfo_children():
            # 创建小粒子
            if random.random() < 0.7:  # 70%的概率添加粒子
                size = random.randint(2, 4)
                x = random.randint(0, 1000)
                y = random.randint(0, 700)

                # 创建一个小圆点作为粒子
                canvas = tk.Canvas(self.main_frame, width=size, height=size,
                                  bg=self.colors['background'], highlightthickness=0)
                canvas.create_oval(0, 0, size, size, fill=self.colors['primary_light'], outline="")
                canvas.place(x=x, y=y)

                # 粒子动画
                speed = random.uniform(0.5, 2)
                self.particle_animation(canvas, x, y, speed)

        # 继续动画循环
        self.root.after(500, self.animate_particles)

    def particle_animation(self, canvas, x, y, speed):
        """粒子动画效果"""
        if y > -10 and canvas.winfo_exists():
            y -= speed
            canvas.place(x=x, y=y)
            self.root.after(50, lambda: self.particle_animation(canvas, x, y, speed))
        else:
            if canvas.winfo_exists():
                canvas.destroy()

    def update_status(self, message):
        """更新状态栏消息"""
        if hasattr(self, 'status_bar'):  # 确保status_bar已存在
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_bar.config(text=f"{timestamp} | {message}")
            self.root.update_idletasks()

    def process_image_channels(self, img):
        """确保图片是3通道(RGB)格式，与模型输入要求匹配"""
        # 如果是4通道(RGBA)，转换为3通道(RGB)
        if img.mode == 'RGBA':
            return img.convert('RGB')
        # 如果是单通道(灰度图)，转换为3通道
        elif img.mode == 'L':
            return img.convert('RGB')
        # 已经是3通道则直接返回
        elif img.mode == 'RGB':
            return img
        # 其他模式尝试转换为RGB
        else:
            return img.convert('RGB')

    def load_model(self):
        """加载预训练模型 - PyTorch版本"""
        self.update_status("正在加载模型...")
        try:
            # 创建模型
            self.model = EfficientNetClassifier(num_classes=self.num_classes)

            # 加载标准 PyTorch 模型权重
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict, strict=True)
                self.update_status("✅ 模型加载成功!")
            else:
                self.update_status(f"❌ 模型文件不存在: {self.model_path}")
                messagebox.showerror("错误", f"模型文件不存在: {self.model_path}\n请先运行 convert_final.py 转换检查点")

            # 将模型移到目标设备
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式

            # 定义图像预处理
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            self.model_loaded = True
            self.update_status("模型加载成功！")

        except Exception as e:
            self.update_status(f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    def load_class_names(self):
        """加载类别名称"""
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            self.update_status(f"已加载 {len(self.class_names)} 个类别名称")
        else:
            messagebox.showerror("错误", "未找到类别名称文件")

    def load_unlocked_animals(self):
        """加载已解锁的动物"""
        try:
            if os.path.exists('unlocked_animals.json'):
                with open('unlocked_animals.json', 'r') as f:
                    self.unlocked_animals = set(json.load(f))
                self.update_status(f"已加载 {len(self.unlocked_animals)} 个已解锁动物")
        except Exception as e:
            self.update_status(f"加载已解锁动物失败: {str(e)}")
            self.unlocked_animals = set()

    def save_unlocked_animals(self):
        """保存已解锁的动物"""
        try:
            with open('unlocked_animals.json', 'w') as f:
                json.dump(list(self.unlocked_animals), f)
            self.update_status(f"已保存 {len(self.unlocked_animals)} 个已解锁动物")
        except Exception as e:
            self.update_status(f"保存已解锁动物失败: {str(e)}")

    def load_user_stats(self):
        """加载用户统计信息"""
        try:
            if os.path.exists('user_stats.json'):
                with open('user_stats.json', 'r') as f:
                    self.user_stats = json.load(f)
        except Exception as e:
            self.update_status(f"加载用户统计失败: {str(e)}")

    def save_user_stats(self):
        """保存用户统计信息"""
        try:
            self.user_stats["animals_unlocked"] = len(self.unlocked_animals)
            self.user_stats["last_played"] = datetime.now().isoformat()

            with open('user_stats.json', 'w') as f:
                json.dump(self.user_stats, f)
        except Exception as e:
            self.update_status(f"保存用户统计失败: {str(e)}")

    def clear_frame(self):
        """清除当前页面 - 添加淡出动画效果"""
        for widget in self.main_frame.winfo_children():
            try:
                # 尝试添加淡出效果
                self.fade_out(widget)
            except:
                # 如果不支持透明度，直接销毁
                widget.destroy()

    def back_to_main(self):
        """返回主页面 - 添加过渡动画"""
        self.clear_frame()
        # 短暂延迟后显示主页面，使过渡更平滑
        self.root.after(300, self.create_main_ui)
        self.update_status("返回主页面")

    def show_animal_recognition(self):
        """显示动物识别页面 - 更现代的布局"""
        self.clear_frame()
        self.update_status("进入动物识别模式")

        # 返回按钮
        back_button = ttk.Button(self.main_frame, text="← 返回主页", command=self.back_to_main, style="Normal.TButton")
        back_button.pack(anchor=tk.NW, padx=10, pady=10)
        back_button.bind("<Enter>", lambda e, b=back_button: b.config(style="Normal.Hover.TButton"))
        back_button.bind("<Leave>", lambda e, b=back_button: b.config(style="Normal.TButton"))

        # 标题区域
        title_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        title_frame.pack(pady=15)

        title_label = ttk.Label(title_frame, text="🐾 动物识别", font=("Segoe UI", 24, "bold"), style="Title.TLabel")
        title_label.pack()

        # 标题下方的装饰线
        separator = ttk.Separator(title_frame, orient="horizontal")
        separator.pack(fill=tk.X, padx=100, pady=10)

        # 创建主内容框架，使用网格布局
        content_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 左侧图片区域
        left_frame = ttk.Frame(content_frame, style="Main.TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        # 图片上传区域卡片，带阴影效果
        upload_frame = ttk.Frame(left_frame, style="Card.TFrame", padding=20)
        upload_frame.pack(fill=tk.BOTH, expand=True)

        # 标题区域
        title_container = ttk.Frame(upload_frame, style="White.TFrame")
        title_container.pack(fill=tk.X, pady=(0, 10))

        upload_title = ttk.Label(title_container, text="🖼️ 图片上传识别",
                                font=("Segoe UI", 16, "bold"), style="White.TLabel")
        upload_title.pack(side=tk.LEFT)

        # 图片显示区域 - 使用Canvas来更好地控制图片显示
        image_display_frame = ttk.Frame(upload_frame, style="Card.TFrame")
        image_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 创建带滚动条的Canvas（如果图片很大）
        self.image_canvas = tk.Canvas(image_display_frame, bg="#f8fafc",
                                      highlightthickness=1,
                                      highlightbackground=self.colors['border'])
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        # 在Canvas中创建图片标签
        self.recognition_image_label = ttk.Label(
            self.image_canvas,
            text="📸 请上传动物图片进行识别\n\n支持格式: JPG, PNG, BMP\n建议尺寸: 456×456 或更大",
            anchor=tk.CENTER,
            font=("Segoe UI", 12),
            style="White.TLabel",
            justify=tk.CENTER
        )
        self.image_canvas.create_window(0, 0, anchor=tk.NW, window=self.recognition_image_label)

        # 绑定Canvas大小变化事件，使图片标签居中
        def center_image(event):
            canvas_width = event.width
            canvas_height = event.height
            self.image_canvas.coords(self.image_canvas.find_all()[0],
                                    canvas_width//2, canvas_height//2)
            self.image_canvas.itemconfig(self.image_canvas.find_all()[0], anchor=tk.CENTER)

        self.image_canvas.bind('<Configure>', center_image)

        # 按钮区域 - 固定在底部，使用独立Frame确保始终可见
        button_separator = ttk.Separator(upload_frame, orient='horizontal')
        button_separator.pack(fill=tk.X, pady=(15, 10))

        button_frame = ttk.Frame(upload_frame, style="White.TFrame")
        button_frame.pack(fill=tk.X, pady=(0, 5))

        # 创建居中的按钮容器
        button_inner = ttk.Frame(button_frame, style="White.TFrame")
        button_inner.pack(expand=True)

        # 上传按钮
        upload_btn = ttk.Button(button_inner, text="📁 上传图片",
                               command=self.upload_image,
                               style="Normal.TButton", width=18)
        upload_btn.pack(side=tk.LEFT, padx=8, pady=5)
        upload_btn.bind("<Enter>", lambda e, b=upload_btn: b.config(style="Normal.Hover.TButton"))
        upload_btn.bind("<Leave>", lambda e, b=upload_btn: b.config(style="Normal.TButton"))

        # 开始识别按钮
        self.start_recognition_btn = ttk.Button(button_inner, text="🔍 开始识别",
                                               command=self.start_recognition,
                                               state=tk.DISABLED,
                                               style="Accent.TButton", width=18)
        self.start_recognition_btn.pack(side=tk.LEFT, padx=8, pady=5)
        self.start_recognition_btn.bind("<Enter>", lambda e, b=self.start_recognition_btn: b.config(style="Accent.Hover.TButton"))
        self.start_recognition_btn.bind("<Leave>", lambda e, b=self.start_recognition_btn: b.config(style="Accent.TButton"))

        # 添加提示信息
        hint_label = ttk.Label(upload_frame,
                              text="💡 提示：上传图片后点击\"开始识别\"按钮进行AI识别",
                              font=("Segoe UI", 9),
                              foreground=self.colors['text_light'],
                              style="White.TLabel")
        hint_label.pack(pady=(5, 0))

        # 右侧结果区域
        right_frame = ttk.Frame(content_frame, style="Main.TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 结果显示区域卡片
        result_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=20)
        result_frame.pack(fill=tk.BOTH, expand=True)

        # 标题区域
        result_title_container = ttk.Frame(result_frame, style="White.TFrame")
        result_title_container.pack(fill=tk.X, pady=(0, 10))

        result_title = ttk.Label(result_title_container, text="🎯 识别结果",
                                font=("Segoe UI", 16, "bold"), style="White.TLabel")
        result_title.pack(side=tk.LEFT)

        # 清空结果按钮
        clear_btn = ttk.Button(result_title_container, text="🗑️ 清空",
                              command=lambda: self.result_text.delete(1.0, tk.END),
                              style="Normal.TButton", width=10)
        clear_btn.pack(side=tk.RIGHT)
        clear_btn.bind("<Enter>", lambda e, b=clear_btn: b.config(style="Normal.Hover.TButton"))
        clear_btn.bind("<Leave>", lambda e, b=clear_btn: b.config(style="Normal.TButton"))

        # 分隔线
        ttk.Separator(result_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))

        # 添加滚动文本框容器
        result_container = ttk.Frame(result_frame, style="White.TFrame")
        result_container.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(result_container, height=15, width=50, yscrollcommand=scrollbar.set,
                                  font=("Segoe UI", 11), wrap=tk.WORD, padx=15, pady=15,
                                  relief=tk.FLAT, borderwidth=0, background="#ffffff",
                                  highlightthickness=1, highlightbackground=self.colors['border'],
                                  highlightcolor=self.colors['primary'])
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)

        # 配置文本标签样式 - 更丰富的样式
        self.result_text.tag_configure("title", font=("Segoe UI", 14, "bold"),
                                      foreground=self.colors['primary_dark'],
                                      spacing1=10, spacing3=10)
        self.result_text.tag_configure("result", font=("Segoe UI", 12),
                                      foreground=self.colors['text'],
                                      spacing1=5)
        self.result_text.tag_configure("highlight", font=("Segoe UI", 13, "bold"),
                                      foreground=self.colors['danger'],
                                      spacing1=8, spacing3=8)
        self.result_text.tag_configure("unlock", font=("Segoe UI", 11, "italic"),
                                      foreground="#2ecc71",
                                      spacing1=5)
        self.result_text.tag_configure("info", font=("Segoe UI", 10),
                                      foreground=self.colors['text_light'],
                                      spacing1=3)

        # 进度条框架
        self.progress_frame = ttk.Frame(result_frame, style="White.TFrame")
        self.progress_frame.pack(fill=tk.X, pady=(15, 0))

        # 进度条标签
        self.progress_label = ttk.Label(self.progress_frame, text="正在识别中...",
                                       font=("Segoe UI", 10),
                                       foreground=self.colors['primary'],
                                       style="White.TLabel")
        self.progress_label.pack(pady=(0, 5))

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate',
                                           length=300)
        self.progress_bar.pack(fill=tk.X)

        # 初始隐藏进度条
        self.progress_frame.pack_forget()

    def upload_image(self):
        """上传图片 - 添加预览动画效果"""
        file_path = filedialog.askopenfilename(
            title="选择动物图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if not file_path:
            return

        try:
            # 保存当前图片路径
            self.current_image_path = file_path

            # 打开图片并处理通道
            img = Image.open(file_path)
            self.processed_img = self.process_image_channels(img)  # 处理通道数

            # 显示图片 - 添加淡入效果
            display_img = img.copy()
            # 添加圆角边框效果
            display_img = self.add_rounded_corners(display_img, 20)
            display_img.thumbnail((450, 350))  # 调整显示尺寸
            photo = ImageTk.PhotoImage(display_img)

            # 先清空现有内容
            self.recognition_image_label.configure(image=photo, text="")
            self.recognition_image_label.image = photo

            # 启用开始识别按钮
            self.start_recognition_btn.config(state=tk.NORMAL)

            # 清空结果框
            self.result_text.delete(1.0, tk.END)

            self.update_status(f"已上传图片: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败: {str(e)}")
            self.current_image_path = None
            self.processed_img = None
            self.start_recognition_btn.config(state=tk.DISABLED)

    def add_rounded_corners(self, img, radius):
        """为图片添加圆角效果 - 更平滑的边缘处理"""
        # 创建一个透明掩码
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)

        # 绘制圆角矩形
        draw.rounded_rectangle([(0, 0), img.size], radius, fill=255)

        # 应用掩码
        result = img.copy()
        result.putalpha(mask)

        # 添加轻微的阴影效果
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, (240, 240, 240))
            background.putalpha(mask)
            result = Image.alpha_composite(background, result)

        return result

    def start_recognition(self):
        """开始识别（在新线程中执行以避免界面卡顿）"""
        if not self.current_image_path or self.processed_img is None:
            return

        # 禁用按钮防止重复点击
        self.start_recognition_btn.config(state=tk.DISABLED)

        # 显示进度条
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_bar.start(10)

        # 显示识别中提示
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在识别中，请稍候...（第一次加载模型需要一定时间，请耐心等待哦~ 🌟）\n\n", "title")

        if not self.model_loaded:
            self.result_text.insert(tk.END, "第一次加载模型需要一定时间，请耐心等待哦~ 🌟", "result")

        self.update_status("正在识别图片中的动物...")

        # 更新用户统计
        self.user_stats["total_recognitions"] += 1
        self.save_user_stats()

        # 在新线程中执行识别
        threading.Thread(target=self.perform_recognition, daemon=True).start()

    def perform_recognition(self):
        """执行识别操作 - PyTorch版本"""
        try:
            # 检查模型是否加载
            if self.model is None or not self.model_loaded:
                # 如果模型未加载，尝试重新加载
                self.load_model()
                if self.model is None or not self.model_loaded:
                    self.root.after(0, lambda: self.show_recognition_error("模型加载失败，无法进行识别"))
                    return

            # 模拟处理时间，让进度条可见
            time.sleep(1)

            # 准备模型输入 - PyTorch方式
            img_tensor = self.transform(self.processed_img).unsqueeze(0).to(self.device)  # 添加批次维度并移到设备 [1, 3, 456, 456]

            # 进行预测
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]  # 转换为概率
                predictions = probabilities.cpu().numpy()  # 移回 CPU 用于后续处理

            top3_indices = np.argsort(predictions)[::-1][:3]

            # 准备结果字符串
            result_str = "识别结果:\n\n"
            for i, idx in enumerate(top3_indices):
                class_name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
                confidence = predictions[idx]
                result_str += f"{i+1}. {class_name}: {confidence*100:.2f}%\n"

            # 解锁识别到的动物
            unlock_message = ""
            if len(top3_indices) > 0:
                top_class_idx = top3_indices[0]
                if top_class_idx < len(self.class_names):
                    top_class = self.class_names[top_class_idx]
                    if top_class not in self.unlocked_animals:
                        self.unlocked_animals.add(top_class)
                        self.save_unlocked_animals()
                        self.save_user_stats()
                        unlock_message = f"\n🎉 恭喜！你已解锁新动物: {top_class}"

            # 在主线程中更新UI
            self.root.after(0, lambda: self.show_recognition_result(result_str, unlock_message))

        except Exception as e:
            self.root.after(0, lambda: self.show_recognition_error(f"识别过程出错: {str(e)}"))

    def show_recognition_result(self, result_str, unlock_message):
        """显示识别结果 - 添加淡入动画"""
        # 停止进度条并隐藏
        self.progress_bar.stop()
        self.progress_frame.pack_forget()

        # 清空并插入新内容
        self.result_text.delete(1.0, tk.END)

        # 添加淡入效果
        self.result_text.insert(tk.END, "识别完成！\n\n", "title")
        self.result_text.insert(tk.END, result_str, "result")

        if unlock_message:
            self.result_text.insert(tk.END, unlock_message, "unlock")

        # 重新启用按钮
        self.start_recognition_btn.config(state=tk.NORMAL)
        self.update_status("识别完成")

    def show_recognition_error(self, error_msg):
        """显示识别错误"""
        # 停止进度条并隐藏
        self.progress_bar.stop()
        self.progress_frame.pack_forget()

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "识别失败 😞\n\n", "title")
        self.result_text.insert(tk.END, error_msg, "highlight")
        self.start_recognition_btn.config(state=tk.NORMAL)  # 重新启用按钮
        self.update_status("识别失败")

    def show_animal_game(self):
        """显示动物认识小游戏页面 - 更吸引人的设计"""
        self.clear_frame()
        self.update_status("进入动物认识小游戏")

        # 返回按钮
        back_button = ttk.Button(self.main_frame, text="← 返回主页", command=self.back_to_main, style="Normal.TButton")
        back_button.pack(anchor=tk.NW, padx=10, pady=10)
        back_button.bind("<Enter>", lambda e, b=back_button: b.config(style="Normal.Hover.TButton"))
        back_button.bind("<Leave>", lambda e, b=back_button: b.config(style="Normal.TButton"))

        # 标题区域
        title_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        title_frame.pack(pady=15)

        title_label = ttk.Label(title_frame, text="🎮 动物认识小游戏", font=("Segoe UI", 24, "bold"), style="Title.TLabel")
        title_label.pack()

        # 标题下方的装饰线
        separator = ttk.Separator(title_frame, orient="horizontal")
        separator.pack(fill=tk.X, padx=100, pady=10)

        # 游戏说明卡片，带阴影效果
        instruction_frame = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
        instruction_frame.pack(pady=20, padx=100, fill=tk.X)

        # 卡片阴影
        instruction_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        instruction_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        instruction_text = """
        欢迎参加动物认识小游戏！

        游戏规则：系统会随机展示动物图片，你需要从四个选项中选择正确答案。
        每答对一题得1分，答错不扣分。完成后会根据你的得分给予评价。
        """

        instruction_label = ttk.Label(instruction_frame, text=instruction_text,
                                     font=("Segoe UI", 11), style="White.TLabel", justify=tk.LEFT)
        instruction_label.pack()

        # 难度选择区域卡片，带阴影
        difficulty_frame = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
        difficulty_frame.pack(pady=10, padx=100, fill=tk.X)

        # 卡片阴影
        difficulty_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        difficulty_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        difficulty_title = ttk.Label(difficulty_frame, text="选择游戏难度",
                                    font=("Segoe UI", 14, "bold"), style="White.TLabel")
        difficulty_title.pack(pady=(0, 15))

        # 使用更美观的难度选择按钮，添加悬停效果
        button_frame = ttk.Frame(difficulty_frame, style="White.TFrame")
        button_frame.pack()

        # 创建难度按钮并添加悬停效果
        easy_btn = ttk.Button(button_frame, text="简单 (5种动物)",
                            command=lambda: self.start_game(5), style="Accent.TButton")
        easy_btn.pack(side=tk.LEFT, padx=15, pady=10)
        easy_btn.bind("<Enter>", lambda e, b=easy_btn: b.config(style="Accent.Hover.TButton"))
        easy_btn.bind("<Leave>", lambda e, b=easy_btn: b.config(style="Accent.TButton"))

        medium_btn = ttk.Button(button_frame, text="中等 (10种动物)",
                              command=lambda: self.start_game(10), style="Accent.TButton")
        medium_btn.pack(side=tk.LEFT, padx=15, pady=10)
        medium_btn.bind("<Enter>", lambda e, b=medium_btn: b.config(style="Accent.Hover.TButton"))
        medium_btn.bind("<Leave>", lambda e, b=medium_btn: b.config(style="Accent.TButton"))

        hard_btn = ttk.Button(button_frame, text="困难 (20种动物)",
                            command=lambda: self.start_game(20), style="Accent.TButton")
        hard_btn.pack(side=tk.LEFT, padx=15, pady=10)
        hard_btn.bind("<Enter>", lambda e, b=hard_btn: b.config(style="Accent.Hover.TButton"))
        hard_btn.bind("<Leave>", lambda e, b=hard_btn: b.config(style="Accent.TButton"))

    def start_game(self, num_animals):
        """开始游戏"""
        # 随机选择动物
        if len(self.class_names) < num_animals:
            messagebox.showerror("错误", f"动物种类不足，无法选择{num_animals}种动物")
            return

        selected_animals = random.sample(self.class_names, num_animals)
        self.game_animals = selected_animals
        self.current_animal_index = 0
        self.score = 0

        # 创建游戏界面
        self.show_game_question()

    def show_game_question(self):
        """显示游戏问题 - 更精美的布局"""
        # 清除之前的游戏界面
        for widget in self.main_frame.winfo_children():
            if not isinstance(widget, ttk.Button) or widget["text"] != "← 返回主页":
                widget.destroy()

        # 显示当前进度和分数卡片
        progress_frame = ttk.Frame(self.main_frame, style="Card.TFrame", padding=15)
        progress_frame.pack(pady=10, padx=100, fill=tk.X)

        # 卡片阴影
        progress_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        progress_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        progress_text = f"进度: {self.current_animal_index+1}/{len(self.game_animals)} | 分数: {self.score}"
        progress_label = ttk.Label(progress_frame, text=progress_text,
                                  font=("Segoe UI", 12, "bold"), style="White.TLabel")
        progress_label.pack()

        # 当前动物
        current_animal = self.game_animals[self.current_animal_index]

        # 获取动物图片
        animal_dir = os.path.join(self.animal_images_dir, current_animal)
        if os.path.exists(animal_dir):
            image_files = [f for f in os.listdir(animal_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            if image_files:
                random_image = random.choice(image_files)
                image_path = os.path.join(animal_dir, random_image)

                # 显示图片卡片，带阴影和圆角
                image_card = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
                image_card.pack(pady=20, padx=100)

                # 卡片阴影
                image_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
                image_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

                # 显示图片
                try:
                    img = Image.open(image_path)
                    # 添加圆角效果和轻微边框
                    img = self.add_rounded_corners(img, 15)

                    # 设置设置图片图片大小限制，防止过大图片过大
                    max_width = 500
                    max_height = 350  # 减小高度限制，为选项留出空间

                    # 获取原始图片尺寸
                    width, height = img.size

                    # 计算缩放比例
                    if width > max_width or height > max_height:
                        # 计算宽度和高度的缩放比例
                        width_ratio = max_width / width
                        height_ratio = max_height / height

                        # 选择较小的缩放比例以确保图片完全在限制范围内
                        scale_ratio = min(width_ratio, height_ratio)

                        # 计算新尺寸
                        new_width = int(width * scale_ratio)
                        new_height = int(height * scale_ratio)

                        # 缩放图片
                        img = img.resize((new_width, new_height), Image.LANCZOS)

                    photo = ImageTk.PhotoImage(img)

                    image_label = ttk.Label(image_card, image=photo, style="White.TLabel")
                    image_label.image = photo
                    image_label.pack()
                except Exception as e:
                    ttk.Label(image_card, text=f"无法加载图片: {str(e)}", style="White.TLabel").pack(pady=10)
            else:
                error_frame = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
                error_frame.pack(pady=20, padx=100)
                ttk.Label(error_frame, text=f"未找到{current_animal}的图片", style="White.TLabel").pack(pady=10)
        else:
            error_frame = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
            error_frame.pack(pady=20, padx=100)
            ttk.Label(error_frame, text=f"未找到{current_animal}的图片目录", style="White.TLabel").pack(pady=10)

        # 生成选项（一个正确答案和三个错误答案）
        options = [current_animal]
        while len(options) < 4:
            if len(self.class_names) < 4:
                messagebox.showerror("错误", "动物种类不足，无法生成选项")
                return

            wrong_animal = random.choice(self.class_names)
            if wrong_animal != current_animal and wrong_animal not in options:
                options.append(wrong_animal)

        random.shuffle(options)
        self.correct_answer = current_animal

        # 显示选项卡片，带阴影
        options_card = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
        options_card.pack(pady=20, padx=100, fill=tk.X)

        # 卡片阴影
        options_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        options_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        options_title = ttk.Label(options_card, text="请选择正确答案:",
                                 font=("Segoe UI", 12, "bold"), style="White.TLabel")
        options_title.pack(pady=(0, 15))

        options_frame = ttk.Frame(options_card, style="White.TFrame")
        options_frame.pack()

        # 创建选项按钮，添加悬停效果
        for i, option in enumerate(options):
            btn = ttk.Button(options_frame, text=option, width=25,
                            command=lambda o=option: self.check_answer(o),
                            style="Normal.TButton")
            btn.grid(row=i//2, column=i%2, padx=15, pady=10)
            # 添加悬停效果
            btn.bind("<Enter>", lambda e, b=btn: b.config(style="Normal.Hover.TButton"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(style="Normal.TButton"))

    def check_answer(self, selected_option):
        """检查答案 - 添加反馈动画"""
        if selected_option == self.correct_answer:
            self.score += 1
            self.user_stats["correct_guesses"] += 1
            # 解锁动物
            self.unlocked_animals.add(self.correct_answer)
            self.save_unlocked_animals()
            self.save_user_stats()
            messagebox.showinfo("结果", "✅ 回答正确！")
        else:
            messagebox.showerror("结果", f"❌ 回答错误！正确答案是: {self.correct_answer}")

        # 下一题或结束游戏
        self.current_animal_index += 1
        if self.current_animal_index < len(self.game_animals):
            # 添加过渡效果
            self.clear_game_question()
            self.root.after(200, self.show_game_question)
        else:
            self.show_game_result()

    def clear_game_question(self):
        """清除当前游戏问题，为下一题做准备"""
        for widget in self.main_frame.winfo_children():
            if not isinstance(widget, ttk.Button) or widget["text"] != "← 返回主页":
                try:
                    self.fade_out(widget)
                except:
                    widget.destroy()

    def show_game_result(self):
        """显示游戏结果 - 更精美的设计"""
        # 清除游戏界面
        for widget in self.main_frame.winfo_children():
            if not isinstance(widget, ttk.Button) or widget["text"] != "← 返回主页":
                widget.destroy()

        # 计算得分百分比
        percentage = (self.score / len(self.game_animals)) * 100

        # 根据得分给出评价
        if percentage >= 90:
            evaluation = "🎉 太棒了！你是动物专家！"
            color = "#27ae60"
            icon = "⭐️⭐️⭐️⭐️⭐️"
        elif percentage >= 70:
            evaluation = "👍 做得很好！你对动物很了解！"
            color = "#f39c12"
            icon = "⭐️⭐️⭐️⭐️"
        elif percentage >= 50:
            evaluation = "😊 不错！继续学习更多动物知识！"
            color = "#f39c12"
            icon = "⭐️⭐️⭐️"
        else:
            evaluation = "📚 加油！多学习动物知识，下次会更好！"
            color = "#e74c3c"
            icon = "⭐️⭐️"

        # 显示结果卡片，带阴影和装饰
        result_card = ttk.Frame(self.main_frame, style="Card.TFrame", padding=30)
        result_card.pack(pady=50, padx=100, fill=tk.BOTH, expand=True)

        # 卡片阴影
        result_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        result_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        result_label = ttk.Label(result_card,
                                text=f"游戏结束！{icon}\n你的得分是: {self.score}/{len(self.game_animals)}",
                                font=("Segoe UI", 18, "bold"), style="White.TLabel")
        result_label.pack(pady=20)

        evaluation_label = ttk.Label(result_card, text=evaluation,
                                   font=("Segoe UI", 14), foreground=color, style="White.TLabel")
        evaluation_label.pack(pady=10)

        # 解锁的动物数量
        unlocked_count = len(self.unlocked_animals)
        total_count = len(self.class_names)

        # 创建进度条
        progress_frame = ttk.Frame(result_card, style="White.TFrame")
        progress_frame.pack(pady=20, fill=tk.X, padx=50)

        ttk.Label(progress_frame, text="解锁进度:", font=("Segoe UI", 11), style="White.TLabel").pack(anchor=tk.W)

        progress_bar = ttk.Progressbar(progress_frame, maximum=total_count, value=unlocked_count)
        progress_bar.pack(fill=tk.X, pady=5)

        ttk.Label(progress_frame, text=f"{unlocked_count}/{total_count}",
                 font=("Segoe UI", 10), style="White.TLabel").pack(anchor=tk.E)

        # 按钮框架
        button_frame = ttk.Frame(result_card, style="White.TFrame")
        button_frame.pack(pady=30)

        # 再玩一次按钮
        play_again_btn = ttk.Button(button_frame, text="再玩一次", command=self.show_animal_game,
                                  style="Normal.TButton")
        play_again_btn.pack(side=tk.LEFT, padx=10)
        play_again_btn.bind("<Enter>", lambda e, b=play_again_btn: b.config(style="Normal.Hover.TButton"))
        play_again_btn.bind("<Leave>", lambda e, b=play_again_btn: b.config(style="Normal.TButton"))

        # 返回主页按钮
        home_btn = ttk.Button(button_frame, text="返回主页", command=self.back_to_main,
                            style="Accent.TButton")
        home_btn.pack(side=tk.LEFT, padx=10)
        home_btn.bind("<Enter>", lambda e, b=home_btn: b.config(style="Accent.Hover.TButton"))
        home_btn.bind("<Leave>", lambda e, b=home_btn: b.config(style="Accent.TButton"))

        self.update_status("游戏结束")

    def show_virtual_zoo(self):
        """显示动物园页面 - 更精美的网格布局"""
        self.clear_frame()
        self.update_status("进入动物园图鉴中...")

        # 返回按钮
        back_button = ttk.Button(self.main_frame, text="← 返回主页", command=self.back_to_main, style="Normal.TButton")
        back_button.pack(anchor=tk.NW, padx=10, pady=10)
        back_button.bind("<Enter>", lambda e, b=back_button: b.config(style="Normal.Hover.TButton"))
        back_button.bind("<Leave>", lambda e, b=back_button: b.config(style="Normal.TButton"))

        # 标题区域
        title_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        title_frame.pack(pady=15)

        title_label = ttk.Label(title_frame, text="🏞️ 动物园图鉴", font=("Segoe UI", 24, "bold"), style="Title.TLabel")
        title_label.pack()

        # 标题下方的装饰线
        separator = ttk.Separator(title_frame, orient="horizontal")
        separator.pack(fill=tk.X, padx=100, pady=10)

        # 显示解锁进度卡片
        progress_card = ttk.Frame(self.main_frame, style="Card.TFrame", padding=15)
        progress_card.pack(pady=10, padx=100, fill=tk.X)

        # 卡片阴影
        progress_shadow = ttk.Frame(self.main_frame, style="Shadow.TFrame")
        progress_shadow.pack(pady=(0, 20), padx=102, fill=tk.X, ipady=2)

        unlocked_count = len(self.unlocked_animals)
        total_count = len(self.class_names)
        progress_text = f"已解锁: {unlocked_count}/{total_count} 种动物 ({unlocked_count/total_count*100:.1f}%)"
        progress_label = ttk.Label(progress_card, text=progress_text, font=("Segoe UI", 12), style="White.TLabel")
        progress_label.pack()

        # 创建选项卡，带有图标
        notebook = ttk.Notebook(self.main_frame, style="Custom.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 陆地动物选项卡
        land_frame = ttk.Frame(notebook)
        notebook.add(land_frame, text="🐘 陆地动物")
        self.create_zoo_tab(land_frame, self.land_animals)

        # 海洋动物选项卡
        sea_frame = ttk.Frame(notebook)
        notebook.add(sea_frame, text="🐠 海洋动物")
        self.create_zoo_tab(sea_frame, self.sea_animals)

        # 空中动物选项卡
        air_frame = ttk.Frame(notebook)
        notebook.add(air_frame, text="🦅 空中动物")
        self.create_zoo_tab(air_frame, self.air_animals)

        # 全部动物选项卡
        all_frame = ttk.Frame(notebook)
        notebook.add(all_frame, text="🐾 全部动物")
        self.create_zoo_tab(all_frame, self.class_names)

    def create_zoo_tab(self, parent, animals):
        """创建动物园选项卡内容 - 更精美的卡片设计"""
        # 创建画布和滚动条
        canvas = tk.Canvas(parent, bg=self.colors['background'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Main.TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set, bg=self.colors['background'])

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 创建一个容器框架来居中内容
        container_frame = ttk.Frame(scrollable_frame, style="Main.TFrame")
        container_frame.pack(expand=True, fill=tk.BOTH)

        # 创建一个框架来放置动物卡片，使其居中
        animals_container = ttk.Frame(container_frame, style="Main.TFrame")
        animals_container.pack(expand=True, anchor=tk.CENTER, padx=20, pady=20)

        # 每行显示6个动物，优化卡片大小和间距
        row, col = 0, 0
        animals_per_row = 5
        animal_frame_size = 250  # 动物框架大小
        image_size = 230  # 图片大小

        for animal in animals:
            # 创建动物框架卡片，带阴影效果
            animal_frame = ttk.Frame(animals_container, style="Card.TFrame", padding=10,
                                    width=animal_frame_size, height=animal_frame_size)
            animal_frame.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
            animal_frame.grid_propagate(False)  # 固定框架大小

            # 添加卡片悬停效果
            animal_frame.bind("<Enter>", lambda e, f=animal_frame: f.configure(style="Hover.TFrame"))
            animal_frame.bind("<Leave>", lambda e, f=animal_frame: f.configure(style="Card.TFrame"))

            # 判断是否已解锁
            is_unlocked = animal in self.unlocked_animals

            # 加载图标
            icon_path = os.path.join(self.zoo_icons_dir, f"{animal}_zoo.png")
            if os.path.exists(icon_path):
                try:
                    img = Image.open(icon_path)
                    if not is_unlocked:
                        # 创建灰色轮廓效果
                        img = img.convert("L")
                        img = ImageOps.autocontrast(img, cutoff=5)
                        img = img.filter(ImageFilter.FIND_EDGES)
                        img = ImageOps.invert(img)
                        background = Image.new('RGB', img.size, (200, 200, 200))
                        img = Image.composite(Image.new('RGB', img.size, (100, 100, 100)), background, img)
                    else:
                        img = img.convert("RGB")

                    # 调整图片尺寸并添加圆角
                    img.thumbnail((image_size, image_size))
                    img = self.add_rounded_corners(img, 10)
                    photo = ImageTk.PhotoImage(img)
                    icon_label = ttk.Label(animal_frame, image=photo, style="White.TLabel")
                    icon_label.image = photo
                    icon_label.pack(pady=5)
                except Exception as e:
                    # 显示占位图标
                    self.create_placeholder_icon(animal_frame, animal, is_unlocked, image_size)
            else:
                # 如果没有图标，显示占位符
                self.create_placeholder_icon(animal_frame, animal, is_unlocked, image_size)

            # 显示动物名称
            name_label = ttk.Label(animal_frame, text=animal if is_unlocked else "???",
                                 font=("Segoe UI", 14, "bold" if is_unlocked else "normal"),
                                 foreground=self.colors['text'] if is_unlocked else self.colors['text_light'],
                                 style="White.TLabel")
            name_label.pack(pady=5)

            # 显示解锁状态
            status_text = "已解锁" if is_unlocked else "未解锁"
            status_color = "#27ae60" if is_unlocked else "#e74c3c"
            status_label = ttk.Label(animal_frame, text=status_text,
                                   font=("Segoe UI", 12),
                                   foreground=status_color,
                                   style="White.TLabel")
            status_label.pack()

            # 更新行列
            col += 1
            if col >= animals_per_row:
                col = 0
                row += 1

        # 配置网格权重，使内容居中
        for i in range(animals_per_row):
            animals_container.columnconfigure(i, weight=1)
        for i in range(row + 1):
            animals_container.rowconfigure(i, weight=1)

    def create_placeholder_icon(self, parent, animal, is_unlocked, size):
        """创建更精美的占位图标"""
        placeholder = Image.new('RGB', (size, size), (240, 240, 240) if is_unlocked else (200, 200, 200))
        draw = ImageDraw.Draw(placeholder)

        # 添加圆角背景
        draw.rounded_rectangle([(10, 10), (size-10, size-10)], 15, fill=(220, 220, 220) if is_unlocked else (180, 180, 180))

        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            try:
                font = ImageFont.truetype("Arial", 40)
            except:
                font = ImageFont.load_default()

        # 绘制动物名称首字母或问号
        if is_unlocked:
            text = animal[0].upper() if animal else "?"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            position = ((size - text_width) // 2, (size - text_height) // 2)
            draw.text(position, text, fill=self.colors['text'] if is_unlocked else self.colors['text_light'], font=font)
        else:
            text = "?"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            position = ((size - text_width) // 2, (size - text_height) // 2)
            draw.text(position, text, fill=(150, 150, 150), font=font)

        # 添加圆角
        placeholder = self.add_rounded_corners(placeholder, 10)

        photo = ImageTk.PhotoImage(placeholder)
        icon_label = ttk.Label(parent, image=photo, style="White.TLabel")
        icon_label.image = photo
        icon_label.pack(pady=5)

# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalRecognitionApp(root)
    root.mainloop()
