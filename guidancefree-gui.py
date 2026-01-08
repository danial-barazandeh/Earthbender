#!/usr/bin/env python
from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL, ControlNetModel, UNet2DConditionModel, DDIMScheduler
)
from torchvision import transforms
from tqdm import tqdm
from skimage.io import imsave
from PIL import Image
import random

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, 
    QVBoxLayout, QLabel, QToolBar, QSpinBox, QStyle,
    QDialog, QPushButton, QSlider, QHBoxLayout, QGridLayout
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QTabletEvent, QAction, QIcon, QImage, QCursor
)
from PyQt6.QtCore import Qt, QSize, QPointF, QThread, pyqtSignal

# --- Professional Stylesheet (QSS) for a modern look ---
PROFESSIONAL_STYLESHEET = """
    QMainWindow, QWidget, QDialog {
        background-color: #2D2D30;
        color: #F1F1F1;
        font-family: Segoe UI;
    }
    QToolBar {
        background-color: #3E3E42;
        border: none;
        padding: 5px;
        spacing: 5px;
    }
    QToolButton, QPushButton {
        background-color: #3E3E42;
        color: #F1F1F1;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 8px 12px;
    }
    QPushButton {
        min-width: 80px;
    }
    QPushButton:disabled {
        background-color: #2D2D30;
        color: #555555;
    }
    QToolButton:hover, QPushButton:hover:!disabled {
        background-color: #4F4F53;
    }
    QPushButton:pressed:!disabled {
        background-color: #007ACC;
    }
    QToolButton:pressed, QToolButton:checked {
        background-color: #007ACC;
        border: 1px solid #007ACC;
    }
    QSpinBox, QLabel {
        background-color: transparent;
        color: #F1F1F1;
        padding: 2px;
    }
    QSpinBox {
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 5px;
    }
    QSlider::groove:horizontal {
        border: 1px solid #555555;
        height: 8px;
        background: #2D2D30;
        margin: 2px 0;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #007ACC;
        border: 1px solid #007ACC;
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }
    QStatusBar {
        color: #A0A0A0;
    }
    QMessageBox {
        background-color: #3E3E42;
    }
"""

# --- Model and Inference Code ---
@torch.no_grad()
def prepare_control_image(image_path: str, device: torch.device, dtype: torch.dtype,
                          mountain_weight: float, river_weight: float, lake_weight: float,
                          res: int = 512):
    """
    Loads and preprocesses the conditioning image into a 4-channel tensor.
    Returns the control tensor and the individual feature masks for post-processing.
    """
    img_loaded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img_loaded is None: raise FileNotFoundError(f"Could not load image from {image_path}")

    if img_loaded.ndim == 2:
        final_rgb = cv2.cvtColor(img_loaded, cv2.COLOR_GRAY2RGB)
    elif img_loaded.ndim == 3:
        if img_loaded.shape[2] == 3:
            final_rgb = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
        elif img_loaded.shape[2] == 4:
            alpha = img_loaded[:, :, 3, np.newaxis] / 255.0
            bgr = img_loaded[:, :, :3]
            background = np.zeros_like(bgr, dtype=bgr.dtype)
            composited_bgr = (bgr * alpha + background * (1 - alpha)).astype(bgr.dtype)
            final_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
        else: raise ValueError(f"Unsupported channels: {img_loaded.shape[2]}")
    else: raise ValueError(f"Unsupported dimensions: {img_loaded.ndim}")

    cond_np = cv2.resize(final_rgb, (res, res), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(cond_np, cv2.COLOR_RGB2HSV)
    color_ranges = {
        'mountains': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
        'rivers':    {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
        'lakes':     {'lower': np.array([35, 100, 100]), 'upper': np.array([85, 255, 255])}
    }
    masks = {}
    for feature, color_range in color_ranges.items():
        mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        masks[feature] = mask.astype(np.float32) / 255.0

    gray = cv2.cvtColor(cond_np, cv2.COLOR_RGB2GRAY)
    structural_edges = cv2.Canny(gray, 50, 150)
    mountain_edges = cv2.Canny((masks['mountains'] * 255).astype(np.uint8), 100, 200)
    river_edges = cv2.Canny((masks['rivers'] * 255).astype(np.uint8), 100, 200)
    lake_edges = cv2.Canny((masks['lakes'] * 255).astype(np.uint8), 100, 200)
    combined_edges = np.maximum.reduce([structural_edges, mountain_edges, river_edges, lake_edges])
    edges = combined_edges.astype(np.float32) / 255.0

    weighted_mountains_mask = masks['mountains'] * mountain_weight
    weighted_rivers_mask = masks['rivers'] * river_weight
    weighted_lakes_mask = masks['lakes'] * lake_weight

    control_features = np.stack([
        weighted_mountains_mask, weighted_rivers_mask, weighted_lakes_mask, edges
    ], axis=0)
    
    control_tensor = torch.from_numpy(control_features).unsqueeze(0).to(device=device, dtype=dtype)
    return control_tensor, masks

@torch.no_grad()
def decode_image(vae, latents):
    """Decodes latents into a PIL image."""
    latents = latents / vae.config.scaling_factor
    image_tensor = vae.decode(latents, return_dict=False)[0]
    image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2
    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
    image_np = (image_tensor * 255).round().astype(np.uint8)
    return Image.fromarray(image_np[0])

class ModelLoaderThread(QThread):
    """Loads all models in a background thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            
            self.progress.emit(10, "Loading VAE and UNet...")
            base_model_id = 'runwayml/stable-diffusion-v1-5'
            vae = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae', torch_dtype=dtype).to(device)
            unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder='unet', torch_dtype=dtype).to(device)
            
            self.progress.emit(50, "Loading ControlNet...")
            controlnet_path = self.model_dir
            if (self.model_dir / "controlnet").exists():
                controlnet_path = self.model_dir / "controlnet"
            controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype).to(device)
            
            vae.eval(); unet.eval(); controlnet.eval()
            
            self.progress.emit(90, "Initializing Scheduler...")
            scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
            
            models = {'vae': vae, 'unet': unet, 'controlnet': controlnet, 'scheduler': scheduler, 'device': device, 'dtype': dtype}
            self.progress.emit(100, "Models loaded!")
            self.finished.emit(models)
        except Exception as e:
            import traceback
            self.error.emit(f"Failed to load models:\n{e}\n\n{traceback.format_exc()}")

class InferenceThread(QThread):
    """Runs only the inference part using pre-loaded models."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(np.ndarray, dict) # Emits raw image and masks
    error = pyqtSignal(str)

    def __init__(self, config, models):
        super().__init__()
        self.config = config
        self.models = models

    def run(self):
        try:
            cfg = self.config
            vae, unet, controlnet = self.models['vae'], self.models['unet'], self.models['controlnet']
            scheduler, device, dtype = self.models['scheduler'], self.models['device'], self.models['dtype']
            
            self.progress.emit(5, "Preparing control image...")
            control_tensor, masks = prepare_control_image(
                cfg['image_path'], device, dtype,
                cfg['mountain_weight'], cfg['river_weight'], cfg['lake_weight']
            )
            
            prompt_embeds = torch.zeros((1, 77, unet.config.cross_attention_dim), device=device, dtype=dtype)
            
            self.progress.emit(15, "Initializing latents...")
            generator = torch.Generator(device=device)
            if cfg['seed']: generator.manual_seed(cfg['seed'])
            else: generator.seed()

            latents = torch.randn((1, unet.config.in_channels, 512 // 8, 512 // 8),
                device=device, generator=generator, dtype=dtype
            ) * scheduler.init_noise_sigma

            scheduler.set_timesteps(cfg['num_inference_steps'], device=device)

            self.progress.emit(20, "Starting diffusion process...")
            for i, t in enumerate(tqdm(scheduler.timesteps, disable=True)):
                latent_model_input = scheduler.scale_model_input(latents, t)
                down_samples, mid_sample = controlnet(
                    latent_model_input, t, prompt_embeds, control_tensor,
                    conditioning_scale=cfg['control_scale'], return_dict=False
                )
                noise_pred = unet(
                    latent_model_input, t, prompt_embeds,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample
                self.progress.emit(20 + int(i / len(scheduler.timesteps) * 75), f"Denoising step {i+1}/{len(scheduler.timesteps)}")

            self.progress.emit(95, "Decoding final image...")
            pil_image_rgb = decode_image(vae, latents)
            pil_image_gray = pil_image_rgb.convert('L')
            
            output_image_np = np.array(pil_image_gray)
            self.progress.emit(100, "Done!")
            self.finished.emit(output_image_np, masks)

        except Exception as e:
            import traceback
            self.error.emit(f"An error occurred during inference:\n{e}\n\n{traceback.format_exc()}")
        finally:
            if 'image_path' in cfg and os.path.exists(cfg['image_path']):
                os.remove(cfg['image_path'])

# --- GUI Code ---
class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(512, 512)
        self.setTabletTracking(True)
        self.setMouseTracking(True) 
        self.canvas_pixmap = QPixmap(512, 512)
        self.canvas_pixmap.fill(Qt.GlobalColor.white)
        self.last_point = QPointF()
        self.current_color = QColor("#ff0000")
        self.max_brush_size = 20
        self.is_erasing = False
        self.is_spraying = False # New state for spray brush
        self.is_drawing = False
        self.spray_spread = 15
        self.spray_density = 10

    def enterEvent(self, event): self.setCursor(Qt.CursorShape.CrossCursor)
    def leaveEvent(self, event): self.unsetCursor()
    def set_color(self, color: QColor): self.current_color = color
    def set_max_brush_size(self, size: int): self.max_brush_size = size
    def set_eraser(self, is_erasing: bool): self.is_erasing = is_erasing
    def set_spray_mode(self, is_spraying: bool): self.is_spraying = is_spraying
    def set_spray_spread(self, spread: int): self.spray_spread = spread
    def set_spray_density(self, density: int): self.spray_density = density
    def clear_canvas(self): self.canvas_pixmap.fill(Qt.GlobalColor.white); self.update()

    def tabletEvent(self, event: QTabletEvent):
        if event.type() == QTabletEvent.Type.TabletPress:
            self.is_drawing = True
            self.last_point = event.position()
            if self.is_spraying:
                self.paint_spray(event.position(), event.pressure())
        elif event.type() == QTabletEvent.Type.TabletMove and self.is_drawing:
            if self.is_spraying:
                self.paint_spray(event.position(), event.pressure())
            elif event.pressure() > 0:
                painter = QPainter(self.canvas_pixmap)
                self.setup_painter(painter, event.pressure())
                painter.drawLine(self.last_point, event.position())
                painter.end()
                self.last_point = event.position()
                self.update()
        elif event.type() == QTabletEvent.Type.TabletRelease:
            self.is_drawing = False
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.last_point = event.position()
            if self.is_spraying:
                self.paint_spray(event.position(), 0.5)
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            if self.is_spraying:
                self.paint_spray(event.position(), 0.5)
            else:
                painter = QPainter(self.canvas_pixmap)
                self.setup_painter(painter, 0.5)
                painter.drawLine(self.last_point, event.position())
                painter.end()
                self.last_point = event.position()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.is_drawing = False

    def paint_spray(self, pos: QPointF, pressure: float):
        painter = QPainter(self.canvas_pixmap)
        self.setup_painter(painter, pressure)
        for _ in range(self.spray_density):
            offset_x = (random.random() - 0.5) * self.spray_spread * 2
            offset_y = (random.random() - 0.5) * self.spray_spread * 2
            painter.drawPoint(int(pos.x() + offset_x), int(pos.y() + offset_y))
        painter.end()
        self.update()

    def setup_painter(self, painter: QPainter, pressure: float):
        if self.is_erasing:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            pen = QPen(Qt.GlobalColor.white, self.max_brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            brush_size = max(1, self.max_brush_size * pressure) if not self.is_spraying else 1
            pen = QPen(self.current_color, brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    def paintEvent(self, event):
        painter = QPainter(self); painter.drawPixmap(self.rect(), self.canvas_pixmap, self.canvas_pixmap.rect()); painter.end()
    def get_pixmap(self): return self.canvas_pixmap.copy()

class PainterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Earthbender (Guidance-Free)")
        self.setGeometry(100, 100, 1100, 950)
        self.model_dir = ""
        self.output_dir = ""
        self.output_counter = 0
        self.models = None
        self.last_raw_output = None
        self.last_masks = None
        self.model_loader_thread = None
        self.inference_thread = None
        self.color_mountain = QColor("#ff0000"); self.color_river = QColor("#0000ff"); self.color_lake = QColor("#00ff00")
        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel); left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.canvas = Canvas(self); left_layout.addWidget(self.canvas)
        
        controls_widget = QWidget(); controls_layout = QGridLayout(controls_widget)
        
        def create_slider_control(label_text, min_val, max_val, default_val, is_float=True, real_time_update_slot=None):
            label = QLabel(label_text)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default_val)
            if is_float:
                value_label = QLabel(f"{default_val/100.0:.2f}")
                slider.valueChanged.connect(lambda val, lbl=value_label: lbl.setText(f"{val/100.0:.2f}"))
            else:
                value_label = QLabel(f"{default_val}")
                slider.valueChanged.connect(lambda val, lbl=value_label: lbl.setText(f"{val}"))
            
            if real_time_update_slot:
                slider.valueChanged.connect(real_time_update_slot)

            return label, slider, value_label

        cn_label, self.control_scale_slider, cn_val_label = create_slider_control("ControlNet Scale:", 0, 200, 100)
        controls_layout.addWidget(cn_label, 0, 0); controls_layout.addWidget(self.control_scale_slider, 0, 1); controls_layout.addWidget(cn_val_label, 0, 2)
        mt_label, self.mountain_weight_slider, mt_val_label = create_slider_control("Mountain Weight:", 0, 200, 100)
        controls_layout.addWidget(mt_label, 1, 0); controls_layout.addWidget(self.mountain_weight_slider, 1, 1); controls_layout.addWidget(mt_val_label, 1, 2)
        rv_label, self.river_weight_slider, rv_val_label = create_slider_control("River Weight:", 0, 200, 100)
        controls_layout.addWidget(rv_label, 2, 0); controls_layout.addWidget(self.river_weight_slider, 2, 1); controls_layout.addWidget(rv_val_label, 2, 2)
        lk_label, self.lake_weight_slider, lk_val_label = create_slider_control("Lake Weight:", 0, 200, 100)
        controls_layout.addWidget(lk_label, 3, 0); controls_layout.addWidget(self.lake_weight_slider, 3, 1); controls_layout.addWidget(lk_val_label, 3, 2)
        
        left_layout.addWidget(controls_widget)

        post_process_widget = QWidget(); post_process_layout = QGridLayout(post_process_widget)
        post_process_widget.setStyleSheet("margin-top: 15px; border-top: 1px solid #444; padding-top: 10px;");
        
        br_mt_label, self.br_mt_slider, br_mt_val_label = create_slider_control("Mountain Brightness:", -100, 100, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(br_mt_label, 0, 0); post_process_layout.addWidget(self.br_mt_slider, 0, 1); post_process_layout.addWidget(br_mt_val_label, 0, 2)
        br_rv_label, self.br_rv_slider, br_rv_val_label = create_slider_control("River Brightness:", -100, 100, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(br_rv_label, 1, 0); post_process_layout.addWidget(self.br_rv_slider, 1, 1); post_process_layout.addWidget(br_rv_val_label, 1, 2)
        br_lk_label, self.br_lk_slider, br_lk_val_label = create_slider_control("Lake Brightness:", -100, 100, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(br_lk_label, 2, 0); post_process_layout.addWidget(self.br_lk_slider, 2, 1); post_process_layout.addWidget(br_lk_val_label, 2, 2)
        
        bl_mt_label, self.bl_mt_slider, bl_mt_val_label = create_slider_control("Mtn. Outer Blur:", 0, 50, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(bl_mt_label, 3, 0); post_process_layout.addWidget(self.bl_mt_slider, 3, 1); post_process_layout.addWidget(bl_mt_val_label, 3, 2)
        bl_br_mt_label, self.bl_br_mt_slider, bl_br_mt_val_label = create_slider_control("Mtn. Blur Brightness:", -50, 50, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(bl_br_mt_label, 4, 0); post_process_layout.addWidget(self.bl_br_mt_slider, 4, 1); post_process_layout.addWidget(bl_br_mt_val_label, 4, 2)

        bl_rv_label, self.bl_rv_slider, bl_rv_val_label = create_slider_control("River Outer Blur:", 0, 50, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(bl_rv_label, 5, 0); post_process_layout.addWidget(self.bl_rv_slider, 5, 1); post_process_layout.addWidget(bl_rv_val_label, 5, 2)
        bl_br_rv_label, self.bl_br_rv_slider, bl_br_rv_val_label = create_slider_control("River Blur Brightness:", -50, 50, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(bl_br_rv_label, 6, 0); post_process_layout.addWidget(self.bl_br_rv_slider, 6, 1); post_process_layout.addWidget(bl_br_rv_val_label, 6, 2)
        
        bl_lk_label, self.bl_lk_slider, bl_lk_val_label = create_slider_control("Lake Outer Blur:", 0, 50, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(bl_lk_label, 7, 0); post_process_layout.addWidget(self.bl_lk_slider, 7, 1); post_process_layout.addWidget(bl_lk_val_label, 7, 2)
        bl_br_lk_label, self.bl_br_lk_slider, bl_br_lk_val_label = create_slider_control("Lake Blur Brightness:", -50, 50, 0, is_float=False, real_time_update_slot=self.update_display)
        post_process_layout.addWidget(bl_br_lk_label, 8, 0); post_process_layout.addWidget(self.bl_br_lk_slider, 8, 1); post_process_layout.addWidget(bl_br_lk_val_label, 8, 2)

        self.post_process_button = QPushButton("Save Processed Image"); self.post_process_button.clicked.connect(self.save_processed_image)
        self.post_process_button.setEnabled(False)
        post_process_layout.addWidget(self.post_process_button, 9, 1, 1, 2)
        
        left_layout.addWidget(post_process_widget)
        main_layout.addWidget(left_panel)

        self.output_label = QLabel("Output will be shown here"); self.output_label.setFixedSize(512, 512)
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setStyleSheet("border: 1px solid #555555; background-color: #202022; border-radius: 4px;")
        main_layout.addWidget(self.output_label)

        toolbar = QToolBar("Main Toolbar"); self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        toolbar.setMovable(False); toolbar.setIconSize(QSize(24, 24))
        style = self.style()
        icon_model_folder = style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        icon_output_folder = style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        icon_run = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        icon_clear = style.standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        icon_spray = style.standardIcon(QStyle.StandardPixmap.SP_DialogYesButton)

        action_select_model = QAction(icon_model_folder, "Select Model Directory", self); action_select_model.triggered.connect(self.select_model_dir)
        action_set_output = QAction(icon_output_folder, "Set Output Directory", self); action_set_output.triggered.connect(self.set_output_dir)
        self.action_run_inference = QAction(icon_run, "Run Inference", self); self.action_run_inference.triggered.connect(self.run_inference)
        self.action_run_inference.setEnabled(False)
        action_clear = QAction(icon_clear, "Clear Canvas", self); action_clear.triggered.connect(self.canvas.clear_canvas)
        
        toolbar.addAction(action_select_model); toolbar.addAction(action_set_output); toolbar.addAction(self.action_run_inference)
        toolbar.addSeparator(); toolbar.addAction(action_clear); toolbar.addSeparator()

        self.action_mountain = self.create_color_action("Moutains (Red)", self.color_mountain, self.activate_mountain)
        self.action_river = self.create_color_action("Rivers (Blue)", self.color_river, self.activate_river)
        self.action_lake = self.create_color_action("Lakes (Green)", self.color_lake, self.activate_lake)
        self.action_eraser = QAction("Eraser", self); self.action_eraser.setCheckable(True); self.action_eraser.triggered.connect(self.activate_eraser)
        self.action_spray = QAction(icon_spray, "Spray", self); self.action_spray.setCheckable(True); self.action_spray.triggered.connect(self.activate_spray)
        
        toolbar.addAction(self.action_mountain); toolbar.addAction(self.action_river); toolbar.addAction(self.action_lake); toolbar.addAction(self.action_eraser); toolbar.addAction(self.action_spray)
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel("Brush:")); self.size_spinbox = QSpinBox(); self.size_spinbox.setRange(1, 100)
        self.size_spinbox.setValue(20); self.size_spinbox.valueChanged.connect(self.canvas.set_max_brush_size)
        self.canvas.set_max_brush_size(20); toolbar.addWidget(self.size_spinbox)
        
        toolbar.addWidget(QLabel("  Steps:")); self.steps_spinbox = QSpinBox(); self.steps_spinbox.setRange(1, 100)
        self.steps_spinbox.setValue(8); toolbar.addWidget(self.steps_spinbox)

        toolbar.addWidget(QLabel("  Spray Spread:")); self.spray_spread_slider = QSlider(Qt.Orientation.Horizontal); self.spray_spread_slider.setRange(1, 100); self.spray_spread_slider.setValue(15)
        self.spray_spread_slider.valueChanged.connect(self.canvas.set_spray_spread); toolbar.addWidget(self.spray_spread_slider)
        
        toolbar.addWidget(QLabel("  Spray Density:")); self.spray_density_slider = QSlider(Qt.Orientation.Horizontal); self.spray_density_slider.setRange(1, 100); self.spray_density_slider.setValue(10)
        self.spray_density_slider.valueChanged.connect(self.canvas.set_spray_density); toolbar.addWidget(self.spray_density_slider)
        
        self.statusBar().showMessage("Ready. Please select a model directory to begin.")
        self.activate_mountain()

    def create_color_action(self, name, color, slot):
        action = QAction(name, self); pixmap = QPixmap(16, 16); pixmap.fill(color)
        action.setIcon(QIcon(pixmap)); action.setCheckable(True); action.triggered.connect(slot)
        return action

    def activate_color(self, active_action, color):
        self.action_eraser.setChecked(False)
        self.action_spray.setChecked(False)
        for action in [self.action_mountain, self.action_river, self.action_lake]:
            action.setChecked(action == active_action)
        self.canvas.set_color(color)
        self.canvas.set_eraser(False)
        self.canvas.set_spray_mode(False)

    def activate_mountain(self): self.activate_color(self.action_mountain, self.color_mountain)
    def activate_river(self): self.activate_color(self.action_river, self.color_river)
    def activate_lake(self): self.activate_color(self.action_lake, self.color_lake)
    
    def activate_eraser(self):
        if self.action_eraser.isChecked():
            self.action_spray.setChecked(False)
            for action in [self.action_mountain, self.action_river, self.action_lake]:
                action.setChecked(False)
            self.canvas.set_eraser(True)
            self.canvas.set_spray_mode(False)
        else:
            self.canvas.set_eraser(False)
            self.action_mountain.setChecked(True)
            self.canvas.set_color(self.color_mountain)

    def activate_spray(self):
        if self.action_spray.isChecked():
            self.action_eraser.setChecked(False)
            is_color_active = any(a.isChecked() for a in [self.action_mountain, self.action_river, self.action_lake])
            if not is_color_active:
                self.action_mountain.setChecked(True)
                self.canvas.set_color(self.color_mountain)
            self.canvas.set_spray_mode(True)
            self.canvas.set_eraser(False)
        else:
            self.canvas.set_spray_mode(False)

    def select_model_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if not folder: return
        
        self.model_dir = Path(folder)
        self.action_run_inference.setEnabled(False)
        self.post_process_button.setEnabled(False)
        self.statusBar().showMessage("Loading models... Please wait.")
        
        self.model_loader_thread = ModelLoaderThread(self.model_dir)
        self.model_loader_thread.finished.connect(self.on_models_loaded)
        self.model_loader_thread.error.connect(self.on_inference_error)
        self.model_loader_thread.progress.connect(lambda p, msg: self.statusBar().showMessage(f"Loading... {p}%: {msg}"))
        self.model_loader_thread.start()

    def on_models_loaded(self, models):
        self.models = models
        self.action_run_inference.setEnabled(True)
        self.statusBar().showMessage("Models loaded and ready. Please set an output directory.", 10000)

    def set_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder: self.output_dir = folder; self.output_counter = 0; self.statusBar().showMessage(f"Output directory set: {self.output_dir}")

    def run_inference(self):
        if self.models is None:
            QMessageBox.warning(self, "Models Not Loaded", "Please select a model directory first.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Directory Not Set", "Please set an output directory first.")
            return

        temp_path = "temp_control_image.png"
        self.canvas.get_pixmap().save(temp_path, "PNG")

        config = {
            'image_path': temp_path,
            'num_inference_steps': self.steps_spinbox.value(), 
            'seed': 42,
            'control_scale': self.control_scale_slider.value() / 100.0,
            'mountain_weight': self.mountain_weight_slider.value() / 100.0,
            'river_weight': self.river_weight_slider.value() / 100.0,
            'lake_weight': self.lake_weight_slider.value() / 100.0,
        }
        
        self.inference_thread = InferenceThread(config, self.models)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.progress.connect(lambda p, msg: self.statusBar().showMessage(f"Processing... {p}%: {msg}"))
        self.inference_thread.start()
        self.statusBar().showMessage(f"Starting inference...")

    def on_inference_finished(self, raw_image_np, masks):
        self.last_raw_output = raw_image_np
        self.last_masks = masks
        self.post_process_button.setEnabled(True)
        self.statusBar().showMessage("Inference complete. Adjust post-processing settings or save.", 10000)
        self.update_display()

    def get_processed_image(self):
        """Applies brightness and the new distance transform blur."""
        if self.last_raw_output is None: return None

        final_image = self.last_raw_output.copy().astype(np.float32)
        masks = self.last_masks

        # --- 1. Apply Blur Effects ---
        total_blur_adjustment = np.zeros_like(final_image, dtype=np.float32)
        for feature, blur_slider, blur_br_slider in [
            ('mountains', self.bl_mt_slider, self.bl_br_mt_slider),
            ('rivers', self.bl_rv_slider, self.bl_br_rv_slider),
            ('lakes', self.bl_lk_slider, self.bl_br_lk_slider)
        ]:
            blur_width = blur_slider.value()
            blur_brightness = blur_br_slider.value()

            if blur_width > 0 and blur_brightness != 0:
                mask = masks[feature]
                inv_mask = ((1 - mask) * 255).astype(np.uint8)
                dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 3)
                falloff_map = 1.0 - np.clip(dist / blur_width, 0, 1)
                feature_blur_adjustment = falloff_map * blur_brightness
                feature_blur_adjustment *= (1 - mask)
                total_blur_adjustment += feature_blur_adjustment
        
        final_image += total_blur_adjustment

        # --- 2. Apply Core Feature Brightness ---
        for feature, slider in [
            ('mountains', self.br_mt_slider),
            ('rivers', self.br_rv_slider),
            ('lakes', self.br_lk_slider)
        ]:
            brightness = slider.value()
            if brightness != 0:
                mask = masks[feature]
                final_image[mask > 0] += brightness

        # --- 3. Finalize ---
        return np.clip(final_image, 0, 255).astype(np.uint8)

    def update_display(self):
        """Gets the fully processed image and updates the display."""
        if self.last_raw_output is None: return

        final_image_np = self.get_processed_image()
        if final_image_np is None: return

        q_image = QImage(final_image_np.data, final_image_np.shape[1], final_image_np.shape[0], final_image_np.strides[0], QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        if not pixmap.isNull():
            self.output_label.setPixmap(pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def save_processed_image(self):
        """Applies the full post-processing and saves the file."""
        if self.last_raw_output is None:
            QMessageBox.warning(self, "No Image", "Please run inference first to generate an image.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Directory Not Set", "Please set an output directory first to save the result.")
            return

        final_image_np = self.get_processed_image()
        if final_image_np is None: return

        output_filename = f"output_{self.output_counter:03d}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        imsave(output_path, final_image_np)
        self.output_counter += 1
        
        self.statusBar().showMessage(f"Processed image saved to {output_path}", 10000)

    def on_inference_error(self, error_message):
        self.statusBar().showMessage("An error occurred.", 10000)
        self.action_run_inference.setEnabled(True)
        QMessageBox.critical(self, "Error", error_message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(PROFESSIONAL_STYLESHEET)
    window = PainterApp()
    window.show()
    sys.exit(app.exec())
